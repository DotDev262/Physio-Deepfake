{ pkgs, ... }:

{
  packages = [
    pkgs.uv
    pkgs.python311
    pkgs.libGL
    pkgs.glib
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
    pkgs.xorg.libXext
    pkgs.libxkbcommon
    pkgs.wayland
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libffi
    pkgs.openssl
  ];

  # Needed for some python packages like opencv, mediapipe, numpy and tensorflow
  env.LD_LIBRARY_PATH = "${pkgs.libGL}/lib:${pkgs.glib.out}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libxcb}/lib:${pkgs.xorg.libXext}/lib:${pkgs.libxkbcommon}/lib:${pkgs.wayland}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libffi}/lib:${pkgs.openssl}/lib";

  enterShell = ''
    echo "devenv environment loaded"
  '';
}
