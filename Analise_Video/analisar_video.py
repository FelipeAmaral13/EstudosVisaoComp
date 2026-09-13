#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analisador de video com Two-Stream e S3D.

Uso:
    python analisar_video.py --source videoplayback.mp4
    python analisar_video.py --source 0                      # webcam
    python analisar_video.py --source videoplayback.mp4 --model mvit
    python analisar_video.py --source videoplayback.mp4 --fps-limit 0
    python analisar_video.py --source videoplayback.mp4 --out painel.mp4 --headless
"""

import argparse
import threading
import time
from collections import deque
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as tvv
from torchvision import models

CLIP_LEN = 16


TILE = 224                  # lado de cada tile da grade
PRED_HEIGHT = 190           # faixa de predicao
FOOTER_HEIGHT = 30
HEADER_HEIGHT = 20          # faixa de titulo dentro de um tile
PRED_BAR_X = 265            # onde comecam as barras de probabilidade
PRED_ROW_H = 26
STATS_ROW_H = 34
CLIP_WARN = 0.05            # fracao de pixels saturados que acende o aviso

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (0, 255, 255)
GREEN = (0, 230, 100)
GREY = (150, 150, 150)
RED = (90, 90, 255)


class ModelSpec(NamedTuple):
    """Especificacao de um modelo de video.
    """
    ctor: str
    weights: str


MODELS = {
    "s3d": ModelSpec("s3d", "S3D_Weights"),
    "r3d": ModelSpec("r3d_18", "R3D_18_Weights"),
    "mvit": ModelSpec("mvit_v2_s", "MViT_V2_S_Weights"),
}

class TwoStreamNetwork(nn.Module):
    """Duas ResNet-50 paralelas: aparencia (RGB) e movimento (fluxo empilhado).
    
    fusion='concat' -> concatena os 2048+2048 descritores e classifica com FC.
    fusion='avg'    -> cada stream tem seu proprio FC; media dos softmax; fusão tardia
    """

    def __init__(self, num_classes=101, flow_stack_size=10,
                 pretrained=True, fusion="concat", dropout=0.5):
        super().__init__()
        self.fusion = fusion
        self.flow_stack_size = flow_stack_size
        in_ch_flow = 2 * flow_stack_size

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        # Stream espacial: 1 frame RGB (3,224,224) -> descritor de 2048 dims.
        # Trocar a fc por Identity transforma a ResNet em extrator de features.
        self.spatial_stream = models.resnet50(weights=weights)
        n_feat = self.spatial_stream.fc.in_features
        self.spatial_stream.fc = nn.Identity()

        # Stream temporal: pilha de K campos de fluxo = (2K,224,224).
        self.temporal_stream = models.resnet50(weights=weights)
        old_conv = self.temporal_stream.conv1
        new_conv = nn.Conv2d(
            in_channels=in_ch_flow,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # [1] Cross-modality pre-training. Sem isto, o stream temporal parte
        # do zero e o pretreino ImageNet 
        if pretrained:
            with torch.no_grad():
                w = old_conv.weight.data                       # [64, 3, 7, 7]
                mean_w = w.mean(dim=1, keepdim=True)           # [64, 1, 7, 7]
                new_conv.weight.copy_(mean_w.repeat(1, in_ch_flow, 1, 1))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias.data)
        self.temporal_stream.conv1 = new_conv
        self.temporal_stream.fc = nn.Identity()

        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(n_feat * 2, num_classes),
            )
        elif fusion == "avg":
            self.fc_spatial = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(n_feat, num_classes))
            self.fc_temporal = nn.Sequential(nn.Dropout(dropout),
                                             nn.Linear(n_feat, num_classes))
        else:
            raise ValueError("fusion deve ser 'concat' ou 'avg'")

    def temporal_conv1_energy(self, x_flow):
        """
        Calcula a energia media dos filtros da primeira camada do stream temporal.
        """
        with torch.no_grad():
            act = self.temporal_stream.conv1(x_flow)   # (1, 64, 112, 112)
        return act[0].abs().mean(dim=0).cpu().numpy()  # energia media

    def forward(self, x_rgb, x_flow, return_streams=False):
        f_s = self.spatial_stream(x_rgb)
        f_t = self.temporal_stream(x_flow)

        if self.fusion == "concat":
            # Fusao precoce: a FC aprende a pesar aparencia vs movimento.
            logits = self.classifier(torch.cat((f_s, f_t), dim=1))
            probs = F.softmax(logits, dim=1)
            parts = (probs, probs)
        else:
            # Fusao tardia: cada stream vota com seu proprio softmax.
            p_s = F.softmax(self.fc_spatial(f_s), dim=1)
            p_t = F.softmax(self.fc_temporal(f_t), dim=1)
            probs = (p_s + p_t) / 2.0
            parts = (p_s, p_t)

        return (probs, *parts) if return_streams else probs



# 2. Fluxo optico: a representacao de movimento do Two-Stream
class FlowExtractor:
    """Calcula fluxo denso em resolucao reduzida e devolve o campo ja
    reescalado para a grade da rede (crop_size), normalizado em [-1, 1].

    Etapas (visiveis no painel):
        frame BGR -> cinza reduzido -> fluxo (dx,dy) -> resize + reescala
                  -> clip em +-bound -> divisao por bound  ->  [-1, 1]
    """

    def __init__(self, backend="dis", proc_width=256, crop_size=224, bound=20.0):
        self.crop = crop_size
        self.bound = bound
        self.proc_width = proc_width
        self.backend = backend
        self.prev = None
        self.last_stats = {}
        if backend == "dis":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        elif backend != "farneback":
            raise ValueError("backend deve ser 'dis' ou 'farneback'")

    def _to_gray(self, frame):
        h, w = frame.shape[:2]
        nh = max(int(round(h * self.proc_width / w)), 16)
        small = cv2.resize(frame, (self.proc_width, nh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def __call__(self, frame):
        gray = self._to_gray(frame)
        if self.prev is None:
            self.prev = gray
            return None

        if self.backend == "dis":
            flow = self.dis.calc(self.prev, gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = gray
        raw_mag = float(np.mean(np.linalg.norm(flow, axis=2)))

        h, w = flow.shape[:2]
        flow = cv2.resize(flow, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= self.crop / float(w)
        flow[..., 1] *= self.crop / float(h)

        clipped = float(np.mean(np.abs(flow) > self.bound))

        np.clip(flow, -self.bound, self.bound, out=flow)

        self.last_stats = {"raw_mag": raw_mag, "proc_res": f"{w}x{h}",
                           "clipped": clipped}
        return flow / self.bound


def stack_flow(buffer):
    """[T x (H,W,2)] -> tensor (1, 2T, H, W) intercalado x1,y1,x2,y2,..."""
    stacked = np.concatenate(list(buffer), axis=2)          # (H, W, 2T)
    return torch.from_numpy(stacked.transpose(2, 0, 1)).unsqueeze(0).float()



# 3. Modelo pre-treinado: a predicao que realmente funciona

def load_model(key, device):
    """Carrega o modelo, o transform oficial dos pesos e os metadados.

    Usar weights.transforms() em vez de normalizar na mao evita o erro mais
    comum aqui: cada modelo de video tem media/desvio proprios (S3D usa
    0.45/0.225, nao os valores do ImageNet).
    """
    spec = MODELS[key]
    weights = getattr(tvv, spec.weights).DEFAULT
    model = getattr(tvv, spec.ctor)(weights=weights).to(device).eval()
    # A acuracia vem dos proprios pesos em vez de ser transcrita na mao, que
    # e uma fonte classica de numero desatualizado.
    acc = weights.meta["_metrics"]["Kinetics-400"]["acc@1"]
    return model, weights.transforms(), weights.meta["categories"], acc


def make_clip(frames, transform, device):
    """Lista de frames BGR -> tensor (1, 3, T, H, W) pronto para a rede.

    O transform do torchvision espera (T, C, H, W) em uint8; ele cuida do
    resize, crop central e normalizacao.
    """
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    clip = torch.from_numpy(np.stack(rgb))          # (T, H, W, C)
    clip = clip.permute(0, 3, 1, 2)                 # (T, C, H, W)
    return transform(clip).unsqueeze(0).to(device)  # (1, C, T, H, W)


def top_k(probs, k):
    """Indices das k classes mais provaveis, da maior para a menor."""
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]



# 4. Visualizacoes

def flow_to_color(flow):
    """Codifica o campo vetorial em HSV: matiz = direcao, brilho = magnitude.

    E a convencao classica de visualizacao de fluxo optico (Middlebury):
    olhando a cor voce le para onde cada regiao esta se movendo, e o brilho
    diz o quao rapido.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)   # direcao -> matiz
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def component_to_gray(channel, gain=1.0):
    """Um canal do fluxo (dx ou dy, em [-1,1]) como imagem cinza.

    Cinza medio = parado; claro = deslocamento positivo; escuro = negativo.
    Esta e literalmente a aparencia de um dos 2K canais que entram na conv1
    do stream temporal.

    gain so amplifica a EXIBICAO: com bound alto uma cena comum ocupa poucos
    porcento da faixa e o tile sairia cinza uniforme.
    """
    img = ((channel * gain + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def activation_to_color(energy):
    """Mapa de energia da conv1 temporal como imagem colorida (INFERNO).

    Escala por percentil em vez do maximo: um unico pixel ruidoso comprime
    todo o resto do mapa para perto de zero.
    """
    hi = float(np.percentile(energy, 99)) or 1.0
    norm = (energy / hi * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def label_tile(img, title, size=TILE):
    """Padroniza um tile do painel e escreve o titulo da etapa."""
    tile = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(tile, (0, 0), (size, HEADER_HEIGHT), (0, 0, 0), -1)
    cv2.putText(tile, title, (5, 15), FONT, 0.45, CYAN, 1, cv2.LINE_AA)
    return tile


def draw_flow_stats(size, stats, n_fields):
    """Tile com os numeros do fluxo optico que normalmente ficam invisiveis."""
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (size, HEADER_HEIGHT), (0, 0, 0), -1)
    cv2.putText(panel, "5. estado do fluxo", (5, 15), FONT, 0.45, CYAN, 1,
                cv2.LINE_AA)

    clipped = stats.get("clipped", 0.0)
    rows = [
        ("movimento medio", f"{stats.get('raw_mag', 0):.2f} px", (200, 200, 200)),
        ("resolucao calc.", stats.get("proc_res", "?"), (200, 200, 200)),
        ("saturado p/ bound", f"{clipped * 100:.1f}%",
         RED if clipped > CLIP_WARN else (140, 220, 140)),
        ("campos na pilha", f"{n_fields} ({n_fields * 2} canais)", (200, 200, 200)),
    ]
    for i, (k, v, color) in enumerate(rows):
        y = 52 + i * STATS_ROW_H
        cv2.putText(panel, k, (12, y), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        cv2.putText(panel, v, (12, y + 18), FONT, 0.52, color, 1, cv2.LINE_AA)

    if clipped > CLIP_WARN:
        cv2.putText(panel, "bound baixo p/ esta cena", (12, size - 12),
                    FONT, 0.38, RED, 1, cv2.LINE_AA)
    return panel


def draw_predictions(width, height, ranking, categories, model_key, latency):
    """Ranking do modelo pre-treinado.

    Mostrar so o argmax esconde a informacao mais util: QUAO apertada foi a
    decisao. Duas classes empatadas em 30% e uma isolada em 60% viram o mesmo
    rotulo, mas significam coisas diferentes sobre a confianca do modelo.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, 22), (35, 35, 35), -1)
    cv2.putText(panel, f"PREDICAO -- {model_key.upper()}", (8, 16),
                FONT, 0.45, CYAN, 1, cv2.LINE_AA)

    if not ranking:
        cv2.putText(panel, f"acumulando clipe de {CLIP_LEN} frames...", (8, 50),
                    FONT, 0.5, GREY, 1, cv2.LINE_AA)
        return panel

    for rank, (idx, prob) in enumerate(ranking):
        y = 34 + rank * PRED_ROW_H
        if y + 18 > height - FOOTER_HEIGHT:
            break
        color = GREEN if rank == 0 else (125, 125, 125)
        cv2.putText(panel, f"{rank + 1}.", (8, y + 14), FONT,
                    0.42, (165, 165, 165), 1, cv2.LINE_AA)
        cv2.putText(panel, categories[idx][:30], (30, y + 14), FONT, 0.44,
                    (240, 240, 240) if rank == 0 else (160, 160, 160), 1,
                    cv2.LINE_AA)
        bx = PRED_BAR_X
        bw = width - bx - 58
        cv2.rectangle(panel, (bx, y + 2), (bx + bw, y + 16), (52, 52, 52), -1)
        cv2.rectangle(panel, (bx, y + 2), (bx + int(bw * prob), y + 16), color, -1)
        cv2.putText(panel, f"{prob * 100:5.1f}%", (bx + bw + 6, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (215, 215, 215), 1, cv2.LINE_AA)

    margin = ranking[0][1] - (ranking[1][1] if len(ranking) > 1 else 0.0)
    cv2.rectangle(panel, (0, height - 26), (width, height), (28, 28, 28), -1)
    cv2.putText(panel, f"margem 1o-2o: {margin * 100:.1f}%   |   "
                       f"latencia: {latency * 1000:.0f} ms/clipe",
                (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (140, 200, 140), 1, cv2.LINE_AA)
    return panel


def build_panel(frame, flow_buf, stats, activation, ranking, categories,
                model_key, latency):
    """Grade 3x2 com o pipeline Two-Stream + faixa de predicao embaixo."""
    newest = flow_buf[-1]
    # O ganho e so de exibicao: com bound alto uma cena comum ocupa poucos
    # porcento da faixa e os tiles sairiam cinza uniforme.
    peak = float(np.abs(newest).max())
    gain = min(1.0 / peak, 40.0) if peak > 1e-6 else 1.0

    act_tile = (label_tile(activation_to_color(activation),
                           "4. conv1 do stream temporal")
                if activation is not None
                else label_tile(np.zeros((TILE, TILE, 3), np.uint8),
                                "4. conv1 (aguardando)"))

    tiles = [
        label_tile(frame, "1. frame de entrada"),
        label_tile(flow_to_color(newest), "2. fluxo (cor=direcao)"),
        label_tile(component_to_gray(newest[..., 0], gain),
                   f"3a. canal dx (x{gain:.0f})"),
        label_tile(component_to_gray(newest[..., 1], gain),
                   f"3b. canal dy (x{gain:.0f})"),
        act_tile,
        draw_flow_stats(TILE, stats, len(flow_buf)),
    ]
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    pred = draw_predictions(grid.shape[1], PRED_HEIGHT, ranking, categories,
                            model_key, latency)

    footer = np.zeros((FOOTER_HEIGHT, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, "tiles 1-4: Two-Stream (2014), movimento pre-computado "
                        " |  predicao: 3D-CNN aprende o tempo sozinha",
                (10, 20), FONT, 0.42, GREY, 1, cv2.LINE_AA)
    return np.vstack([grid, pred, footer])



# 5. Politica de inferencia 
def should_infer(clip_size, clip_version, last_version, frame_idx, stride):
    """Decide se vale rodar a rede neste frame.

    Exige tres condicoes, e a do meio e a que corrige um desperdicio real:
      1. o clipe esta completo;
      2. ele MUDOU desde a ultima inferencia -- sem isso, um --stride menor
         que --sample faz a rede reprocessar um clipe identico, gastando
         centenas de ms para chegar exatamente ao mesmo resultado;
      3. o stride de frames foi respeitado.
    """
    return (clip_size == CLIP_LEN
            and clip_version != last_version
            and frame_idx % stride == 0)


class InferenceWorker:
    """Roda a rede em uma thread separada para o video nao travar.

    O problema que isto resolve: a inferencia custa ~530 ms em CPU. Feita
    dentro do loop, ela congela a exibicao por meio segundo a cada disparo --
    o video anda em solavancos. Aqui a thread principal so captura, desenha e
    exibe (~8 ms/frame), enquanto o worker processa o clipe mais recente em
    paralelo e publica o resultado quando termina.

    A consequencia honesta e que o rotulo exibido fica alguns frames atrasado
    em relacao a imagem. Para leitura humana isso e imperceptivel; o que
    incomoda de verdade e o congelamento.

    Nao ha fila: se um clipe novo chega enquanto o worker trabalha, o anterior
    e descartado. Guardar uma fila so acumularia atraso -- em video ao vivo o
    resultado velho nao interessa.

    A GIL nao atrapalha porque o trabalho pesado esta dentro do PyTorch, que
    libera a GIL durante as operacoes de tensor.
    """

    def __init__(self, model, transform, device, two_stream, topk):
        self._model = model
        self._transform = transform
        self._device = device
        self._two_stream = two_stream
        self._topk = topk

        self._pending = None            # proximo trabalho (clipe, fluxo)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop = False

        # Resultado publicado; lido pela thread principal a cada frame.
        self.ranking = []
        self.latency = 0.0
        self.activation = None
        self.busy = False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, clip_frames, flow_fields):
        """Entrega um novo trabalho, substituindo qualquer um nao iniciado."""
        with self._wake:
            self._pending = (list(clip_frames), list(flow_fields))
            self._wake.notify()

    def _loop(self):
        while True:
            with self._wake:
                while self._pending is None and not self._stop:
                    self._wake.wait()
                if self._stop:
                    return
                clip_frames, flow_fields = self._pending
                self._pending = None
                self.busy = True

            t0 = time.time()
            with torch.no_grad():
                probs = torch.softmax(self._model(
                    make_clip(clip_frames, self._transform, self._device))[0],
                    dim=0)
            ranking = top_k(probs.cpu().numpy(), self._topk)

            activation = None
            if flow_fields:
                activation = self._two_stream.temporal_conv1_energy(
                    stack_flow(flow_fields).to(self._device))

            with self._lock:
                self.ranking = ranking
                self.activation = activation
                self.latency = time.time() - t0
                self.busy = False

    def close(self):
        with self._wake:
            self._stop = True
            self._wake.notify()
        self._thread.join(timeout=2.0)



# 6. Loop principal

def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    print(f"[info] carregando {args.model} em {device}...")
    model, transform, categories, acc = load_model(args.model, device)
    print(f"[info] {len(categories)} classes Kinetics-400 | top-1 {acc:.1f}%")

    # Two-Stream: usado para revelar o que a conv1 temporal enxerga no fluxo.
    # pretrained=True importa aqui -- e o cross-modality init que da sentido
    # ao mapa de ativacao; com pesos aleatorios o tile viraria ruido.
    two_stream = TwoStreamNetwork(flow_stack_size=args.stack,
                                  pretrained=True).to(device).eval()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Nao foi possivel abrir a fonte: {src}")

    flow_ex = FlowExtractor(args.flow, args.proc_width, TILE, args.bound)
    flow_buf = deque(maxlen=args.stack)
    clip_buf = deque(maxlen=CLIP_LEN)

    worker = InferenceWorker(model, transform, device, two_stream, args.topk)

    # Aquecimento: a primeira passada pela rede aloca buffers internos e e
    # bem mais lenta que as seguintes. Fazer isso agora evita que o primeiro
    # engasgo aconteca com o video ja rodando.
    print("[info] aquecendo o modelo...")
    with torch.no_grad():
        blank = [np.zeros((TILE, TILE, 3), np.uint8)] * CLIP_LEN
        model(make_clip(blank, transform, device))

    panel, writer = None, None
    clip_version, last_version, n = 0, -1, 0
    displayed = 0
    frame_period = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    next_frame_at = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Caminho 1: fluxo optico -- o mecanismo visivel no painel.
            flow = flow_ex(frame)
            if flow is not None:
                flow_buf.append(flow)

            # Caminho 2: clipe RGB -- o que alimenta o modelo pre-treinado.
            # Amostrar 1 a cada `sample` faz o clipe cobrir mais tempo real:
            # com sample=2 a 25fps, 16 frames cobrem ~1.3s em vez de 0.6s.
            if n % args.sample == 0:
                clip_buf.append(frame.copy())
                clip_version += 1

            # Despacha para a thread em vez de bloquear aqui. Se o worker
            # ainda estiver ocupado, o frame simplesmente segue: o painel
            # continua mostrando o ultimo resultado valido.
            if (not worker.busy
                    and should_infer(len(clip_buf), clip_version, last_version,
                                     n, args.stride)):
                last_version = clip_version
                worker.submit(clip_buf,
                              flow_buf if len(flow_buf) == args.stack else [])
                if args.verbose and worker.ranking:
                    txt = "  ".join(f"{categories[i]}={p * 100:.1f}%"
                                    for i, p in worker.ranking)
                    print(f"[{n:05d}] {txt}")

            if flow_buf:
                panel = build_panel(frame, flow_buf, flow_ex.last_stats,
                                    worker.activation, worker.ranking,
                                    categories, args.model, worker.latency)

            n += 1
            if panel is None:
                continue

            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                        cap.get(cv2.CAP_PROP_FPS) or 25.0,
                        (panel.shape[1], panel.shape[0]))
                    if not writer.isOpened():
                        raise IOError(f"Nao foi possivel gravar em {args.out}")
                writer.write(panel)

            if not args.headless:
                # Sem a inferencia bloqueando, o loop passa a rodar MAIS rapido
                # que a cadencia natural do video. Este espaçamento devolve o
                # ritmo original em vez de exibir tudo acelerado.
                if frame_period:
                    atraso = next_frame_at - time.time()
                    if atraso > 0:
                        time.sleep(atraso)
                    next_frame_at = max(next_frame_at + frame_period,
                                        time.time())
                cv2.imshow("Two-Stream (mecanismo) + Kinetics (predicao)", panel)
                displayed += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[info] interrompido pelo usuario")
    finally:
        # finally garante que o mp4 receba seu trailer mesmo em Ctrl+C ou
        # excecao -- sem isso o arquivo de saida fica corrompido.
        worker.close()
        cap.release()
        if writer:
            writer.release()
            print(f"[info] video gravado em {args.out}")
        if not args.headless:
            cv2.destroyAllWindows()

    if n == 0:
        print("[aviso] nenhum frame foi lido da fonte")


def build_parser():
    p = argparse.ArgumentParser(
        description="Painel Two-Stream + predicao de modelo pre-treinado")
    p.add_argument("--source", default="0",
                   help="indice da webcam ou caminho do video")
    p.add_argument("--model", choices=list(MODELS), default="s3d",
                   help="s3d (rapido) | r3d | mvit (melhor acuracia)")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--stack", type=int, default=10, help="K campos de fluxo")
    p.add_argument("--flow", choices=["dis", "farneback"], default="dis")
    p.add_argument("--proc-width", type=int, default=256,
                   help="largura de calculo do fluxo (menor = mais rapido)")
    p.add_argument("--bound", type=float, default=4.0, help="clip do fluxo em px")
    p.add_argument("--sample", type=int, default=2,
                   help="pega 1 frame a cada N para montar o clipe")
    p.add_argument("--stride", type=int, default=16,
                   help="1 inferencia a cada N frames")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--fps-limit", type=float, default=25.0,
                   help="cadencia de exibicao (0 = o mais rapido possivel)")
    p.add_argument("--out", help="grava o painel em video")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def parse_args(argv=None):
    """Faz o parse e valida os limites que quebrariam o loop.
    """
    p = build_parser()
    args = p.parse_args(argv)
    for name in ("sample", "stride", "stack", "topk", "threads"):
        if getattr(args, name) < 1:
            p.error(f"--{name} deve ser >= 1 (recebido: {getattr(args, name)})")
    if args.proc_width < 32:
        p.error("--proc-width deve ser >= 32")
    if args.bound <= 0:
        p.error("--bound deve ser > 0")
    return args


if __name__ == "__main__":
    run(parse_args())
