### Script - compilação e instalação do OpenCV na Raspberry Pi Zero W

Neste repositório, você encontra um script completo para compilação e instalação do OpenCV-3.1 

Para utilizar o script, faça o seguinte:

1. Clone este repositório na sua Raspberry PI com o comando: git clone https://github.com/FelipeAmaral13/ProjetosVisaoComp/tree/master/Script_instalacao_openCV-3.1.git
2. Entre no diretório resultante da etapa anterior e execute o script desta forma: sudo ./InstalaOpenCV
3. Aguarde a compilação e instalação serem feitas. Este processo leva algumas horas.
  - Se ocorrer esse erro:
  
  *modules/highgui/CMakeFiles/opencv_highgui.dir/build.make:230: recipe for target 'modules/highgui/CMakeFiles/opencv_highgui.dir/src/cap_ffmpeg.cpp.o' failed
  make[2]: *** [modules/highgui/CMakeFiles/opencv_highgui.dir/src/cap_ffmpeg.cpp.o] Error 1
  CMakeFiles/Makefile2:2349: recipe for target 'modules/highgui/CMakeFiles/opencv_highgui.dir/all' failed*
  
  Faça:
  
  Copie e cole-o na parte superior de:
  
  - **opencv-3.3.0/modules/videoio/src/cap_ffmpeg_impl.hpp**
  
  #define AV_CODEC_FLAG_GLOBAL_HEADER (1 << 22)
  
  #define CODEC_FLAG_GLOBAL_HEADER AV_CODEC_FLAG_GLOBAL_HEADER
  
  #define AVFMT_RAWPICTURE 0x0020

