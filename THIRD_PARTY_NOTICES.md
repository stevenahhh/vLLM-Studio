# Third-Party Notices

This project can integrate with external tools that are not vendored into this
repository. Review their licenses before distributing a bundled build, Docker
image, fork, or modified copy.

## TurboQuant

- Repository: https://github.com/0xSero/turboquant
- Use in this project: optional host-process vLLM wrapper, enabled only when
  `VLLM_TURBOQUANT=1` and TurboQuant is installed separately.
- Upstream license: GNU General Public License v3.0 (`GPL-3.0`)

TurboQuant is not copied into this repository and is not installed by default.
If you distribute a bundle or image that includes TurboQuant, comply with
TurboQuant's GPL-3.0 license terms.

This notice is for project hygiene and is not legal advice.
