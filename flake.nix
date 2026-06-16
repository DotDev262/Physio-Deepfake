{
  description = "Physio-Deepfake Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        # Libraries required by opencv, mediapipe, numpy, etc.
        libPath = with pkgs; lib.makeLibraryPath [
          libGL
          glib.out
          xorg.libX11
          xorg.libxcb
          xorg.libXext
          libxkbcommon
          wayland
          stdenv.cc.cc.lib
          zlib
          libffi
          openssl
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            python311
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
            echo "Physio-Deepfake environment loaded!"
          '';
        };
      }
    );
}
