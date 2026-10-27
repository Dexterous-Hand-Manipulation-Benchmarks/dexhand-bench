{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  }
}:

# Optional NixOS helper: system libraries for the uv-managed Python environment.
# Usage: nix-shell (or `direnv allow` with .envrc), then `uv sync --extra <method>`.
pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
  nativeBuildInputs = with pkgs; [
    gcc
    gnumake
    pkg-config
    cmake
    wget
    curl
    uv
  ];

  buildInputs = with pkgs; [
    glibc.dev
    zlib
    mesa
    cacert
    patchelf
    stdenv.cc.cc.lib
    alsa-lib
    vulkan-loader
    glib

    # Graphics & UI dependencies often needed by cv2 and 3D visualizers
    libGL
    libGLU
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libSM
    xorg.libICE
  ];

  GLIBC_TUNABLES = "glibc.rtld.execstack=2";

  shellHook = ''
    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    export LD_LIBRARY_PATH="${pkgs.glib.out}/lib:/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libGL}/lib:${pkgs.libglvnd}/lib:${pkgs.mesa}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXrender}/lib:${pkgs.xorg.libXext}/lib:$LD_LIBRARY_PATH"
    export MUJOCO_GL=egl

    echo "dexhand-bench nix shell ready: run 'uv sync --extra <flashsac|fasttd3|reppo|ppo>'"
  '';
}