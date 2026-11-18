# Licensing

Kinatric licenses its own software and firmware contributions under **GNU GPL version 3 or any later version (GPL-3.0-or-later)** and its own robot hardware designs under **CERN-OHL-W-2.0**. These grants apply only to rights held by Kinatric and contributors authorized to make the grant. They do not claim ownership of upstream work.

The full, unmodified texts are in [LICENSES/GPL-3.0-or-later.txt](LICENSES/GPL-3.0-or-later.txt) and [LICENSES/CERN-OHL-W-2.0.txt](LICENSES/CERN-OHL-W-2.0.txt). The root LICENSE contains the software license; this document defines its scope.

## Coverage

| Material | Applicable terms |
| --- | --- |
| `source/d2w_rl_lab/`, `scripts/`, build/configuration files and software documentation | GPL-3.0-or-later for Kinatric-owned software and modifications; upstream portions retain their notices. |
| `resources/d2w_description/` | CERN-OHL-W-2.0 for Kinatric-owned robot design files (URDF, meshes and hardware parameters); GPL-3.0-or-later for Kinatric-owned launch/build code. Original embedded third-party notices remain applicable. |
| `source/rsl_rl/`, `docs/licenses/` | Original upstream components/notices retain their terms. Kinatric-owned modifications, if any, are GPL-3.0-or-later; see package LICENSING.md. |
| Checkpoints, exported policies, replay datasets and generated runtime assets | Not distributed in the current source tree. Kinatric-owned weights/checkpoints are GPL-3.0-or-later when released; include artifact notices, provenance and corresponding source. Datasets and third-party/generated assets require their own applicable terms. |

Existing third-party file headers, component LICENSE files and attribution notices remain authoritative for the upstream portions they identify. Kinatric-owned modifications to software are GPL-3.0-or-later, including when combined with compatible permissively licensed upstream code. Preserve those upstream notices as well as the GPL notice when distributing the combined work. An existing upstream SPDX header is not permission to distribute later Kinatric-owned modifications solely under that upstream license. Unmodified third-party components remain under their original terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Hardware grants cover Kinatric-owned CAD, drawings, BOMs, robot-description geometry and hardware parameters in the paths identified above; executable control/training/launch code remains software. Imported designs are not automatically relicensed. Kinatric-owned trained policy weights and checkpoints are GPL-3.0-or-later. No rights to third-party weights, trademarks, logos, photographs, customer uploads or third-party datasets are granted merely because they are stored here or processed by this software. Existing grants for previously distributed revisions remain valid under their terms.

## Commercial use, modification and redistribution

Both selected licenses permit commercial use, modification and redistribution subject to their terms. GPL distributions must include the required notices and corresponding source; installation information is required where GPL section 6 applies. Independent works do not become GPL merely by aggregation. CERN-OHL-W distributions must satisfy the applicable Complete Source and Source Location obligations, including the preferred form for modification and required notices. Qualifying Available Components and External Material retain their own terms. The license texts control.

Hardware Source Location: https://github.com/Kinatric/d2w_rl_lab (use the specific release or commit for the distributed hardware). Before distributing hardware, ensure recipients can obtain the required Complete Source; a private or incomplete repository alone is not sufficient.

## Contributions and release notices

New wholly Kinatric-owned source files should carry `SPDX-License-Identifier: GPL-3.0-or-later` and the actual copyright holder/year. New wholly Kinatric-owned hardware files should carry `SPDX-License-Identifier: CERN-OHL-W-2.0`, or a matching `<filename>.license` sidecar for binary formats. Preserve all original notices in mixed or imported files and identify the license of Kinatric modifications separately; do not replace another author's SPDX identifier or copyright holder. Include these license texts and applicable notices in source and binary packages.

Kinatric-owned trained policies/checkpoints are explicitly GPL-3.0-or-later under this policy. Include an artifact-level license notice, ownership/provenance record and the required corresponding source in each release. Establish the preferred form for modifying the policy, including the editable checkpoint and applicable training/export configuration and scripts; an ONNX export alone must not be assumed sufficient. Licensing training code alone does not automatically license its outputs. Firmware from component vendors retains its vendor terms.

## Published hardware scope

Kinatric publishes its mechanical design exports and robot models under CERN-OHL-W-2.0 where identified in this repository. PCB designs and their source files are proprietary, are not published here, and are excluded from this grant. Native CAD part and assembly files are retained privately; the published mechanical files are STEP exports and other published design files identified by the repository coverage. This scope statement does not amend the license or waive its Complete Source obligations. Software and firmware remain covered separately as described here.
