{ pkgs ? import <nixpkgs> {} }:

let
  conda-fhs = pkgs.buildFHSEnv {
    name = "dexmachina-fhs";

    # These packages will be linked into standard Linux paths (like /lib and /usr/lib)
    # so that pre-compiled Python wheels (like cv2, PyTorch) can find them.
    targetPkgs = pkgs: with pkgs; [
      conda
      glib             # Fixes: libgthread-2.0.so.0
      
      # Graphics & UI dependencies often needed by cv2 and 3D visualizers
      libGL
      libGLU
      libX11
      libXext
      libXrender
      libSM
      libICE
      
      # Standard system libraries
      zlib
      stdenv.cc.cc.lib # Provides libstdc++.so.6
    ];

    # This runs every time you enter the shell
    profile = ''
      # Initialize conda so the 'conda activate' command works immediately
      eval "$(conda shell.bash hook)"
      
      # Optional: Auto-activate your specific environment
      conda-shell 
     
    '';
  };
in
conda-fhs.env
