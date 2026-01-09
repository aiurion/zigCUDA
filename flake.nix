{
  description = "Zig development environment for Linux";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.zig_0_15  # Use Zig 0.15 - upgraded version
            pkgs.curl      # Added for curl in shell
          ];
          shellHook = ''
            echo "Zig development environment ready (using 0.15.2)"
          '';
        };
      }
    );
}