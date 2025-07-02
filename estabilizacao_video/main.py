import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self, input_path, output_path="output_stabilized.mp4", output_size=(640, 480)):
        """
        Inicializa o estabilizador com os parâmetros principais.
        
        Args:
            input_path (str): Caminho do vídeo de entrada.
            output_path (str): Caminho do vídeo de saída estabilizado.
            output_size (tuple): Dimensões finais do vídeo (width, height).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.output_size = output_size
        self.transforms = []
        self.match_frames = []

        # Inicializa leitura do vídeo
        self.cap = cv2.VideoCapture(self.input_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Inicializa o gravador de vídeo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.output_size)

        # Detectores e matchers
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def moving_average(self, curve, radius=5):
        """
        Suaviza um vetor (curva) usando média móvel.
        
        Args:
            curve (np.ndarray): Vetor de movimento (e.g., dx, dy, ângulo).
            radius (int): Raio da janela de suavização.
        
        Returns:
            np.ndarray: Curva suavizada.
        """
        window_size = 2 * radius + 1
        filter_kernel = np.ones(window_size) / window_size
        curve_pad = np.pad(curve, (radius, radius), mode='edge')
        smoothed = np.convolve(curve_pad, filter_kernel, mode='same')
        return smoothed[radius:-radius]

    def smooth_trajectory(self, trajectory):
        """
        Suaviza a trajetória estimada frame a frame (dx, dy, ângulo).
        
        Args:
            trajectory (np.ndarray): Matriz (n_frames, 3) com dx, dy, da.
        
        Returns:
            np.ndarray: Trajetória suavizada.
        """
        smoothed = np.copy(trajectory)
        for i in range(trajectory.shape[1]):
            smoothed[:, i] = self.moving_average(trajectory[:, i])
        return smoothed

    def fix_border(self, frame):
        """
        Aplica zoom leve para eliminar bordas pretas após o warp.
        
        Args:
            frame (np.ndarray): Frame estabilizado.
        
        Returns:
            np.ndarray: Frame corrigido.
        """
        s = frame.shape
        T = cv2.getRotationMatrix2D((s[1] // 2, s[0] // 2), 0, 1.04)
        return cv2.warpAffine(frame, T, (s[1], s[0]))

    def draw_matches(self, img1, kp1, img2, kp2, matches, max_matches=50):
        """
        Desenha os matches entre dois frames.
        
        Args:
            img1, img2 (np.ndarray): Imagens de entrada.
            kp1, kp2 (list): Keypoints detectados.
            matches (list): Matches entre os keypoints.
            max_matches (int): Número máximo de matches a desenhar.
        
        Returns:
            np.ndarray: Imagem com matches desenhados.
        """
        return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def estimate_transforms(self):
        """
        Estima transformações entre pares de frames consecutivos.
        Calcula dx, dy e rotação (da), armazena para reconstrução posterior.
        Também armazena imagens com os matches para visualização.
        """
        _, prev = self.cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(1, C):
            success, curr = self.cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
            kp2, des2 = self.orb.detectAndCompute(curr_gray, None)

            if des1 is None or des2 is None:
                self.transforms.append([0, 0, 0])
                self.match_frames.append(prev)
                continue

            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 10:
                self.transforms.append([0, 0, 0])
                self.match_frames.append(prev)
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            m, _ = cv2.estimateAffinePartial2D(pts1, pts2)
            if m is None:
                self.transforms.append([0, 0, 0])
                self.match_frames.append(prev)
                continue

            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])

            self.transforms.append([dx, dy, da])
            self.match_frames.append(self.draw_matches(prev, kp1, curr, kp2, matches))

            prev_gray = curr_gray
            prev = curr.copy()

    def stabilize(self):
        """
        Pipeline completo de estabilização:
        1. Estima transformações brutas.
        2. Suaviza trajetória.
        3. Aplica transformações suavizadas nos frames.
        4. Gera saída com visualização (original + estabilizado + matches).
        """
        self.estimate_transforms()

        trajectory = np.cumsum(self.transforms, axis=0)
        smoothed = self.smooth_trajectory(trajectory)
        diff = smoothed - trajectory
        new_transforms = np.array(self.transforms) + diff

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i in range(self.n_frames - 1):
            success, frame = self.cap.read()
            if not success:
                break

            dx, dy, da = new_transforms[i]
            m = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da),  np.cos(da), dy]
            ])

            stabilized = cv2.warpAffine(frame, m, (self.w, self.h))
            stabilized = self.fix_border(stabilized)

            match_vis = self.match_frames[i]

            match_resized = cv2.resize(match_vis, (640, 240))
            orig_resized = cv2.resize(frame, (320, 240))
            stab_resized = cv2.resize(stabilized, (320, 240))

            bottom = np.hstack((orig_resized, stab_resized))
            full_frame = np.vstack((match_resized, bottom))

            self.out.write(full_frame)

            cv2.imshow("Match | Original + Estabilizado", full_frame)
            if cv2.waitKey(10) == 27:
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stabilizer = VideoStabilizer("Aporangas.mp4")
    stabilizer.stabilize()