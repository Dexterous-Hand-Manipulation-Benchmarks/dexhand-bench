# Setup

Initialize the conda environment using the dexmachina.yaml file in the dexmachina folder.  
On NixOS you can just load the shell.nix file and the environment will be set up automatically.

## REPPO

To setup REPPO in the dexmachina environment, you have to initialize the REPPO submodule using

```
git submodule update --init --recursive
```

Execute this command in the project root.