{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  }
}:

let
  condaDir = "~/miniconda3";
  envName = "dexmachina";
in
pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
  nativeBuildInputs = with pkgs; [
    gcc
    gnumake
    pkg-config
    cmake
    wget
    curl
  ];

  buildInputs = with pkgs; [
    glibc.dev
    zlib
    libGL
    libGLU
    mesa
    python312
    cacert
    patchelf
    stdenv.cc.cc.lib
    alsa-lib
    vulkan-loader
  ];

  GLIBC_TUNABLES = "glibc.rtld.execstack=2";

  shellHook = ''
    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libGL}/lib:${pkgs.libglvnd}/lib:${pkgs.mesa}/lib:$LD_LIBRARY_PATH"
    export MUJOCO_GL=egl

    # 1. Bootstrap Miniconda globally if missing
    if [ ! -d "${condaDir}" ]; then
      echo "Miniconda not found at ${condaDir}. Downloading..."
      mkdir -p ${condaDir}
      
      if wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; then
        echo "Download complete. Installing Miniconda..."
        bash /tmp/miniconda.sh -b -u -p ${condaDir}
        rm -f /tmp/miniconda.sh
        echo "Miniconda installation finished."
      else
        echo "ERROR: Failed to download Miniconda script." >&2
      fi
    fi

    # 2. Initialize Conda
    if [ -f "${condaDir}/bin/conda" ]; then
      eval "$(${condaDir}/bin/conda shell.bash hook)"
    else
      echo "ERROR: Conda binary not found at ${condaDir}/bin/conda" >&2
      exit 1
    fi

    # 3. Create or update the Conda environment from dexmachina.yaml
    if ! conda info --envs | grep -q "${envName}"; then
      if [ -f "dexmachina/dexmachina.yaml" ]; then
        echo "Creating Conda environment '${envName}' from dexmachina.yaml..."
        conda env create -f dexmachina/dexmachina.yaml
      else
        echo "ERROR: dexmachina.yaml not found!" >&2
        exit 1
      fi
    fi

    # 4. Automatically activate the environment
    conda activate "${envName}"
    echo "Conda environment '${envName}' active and ready!"
  '';
}