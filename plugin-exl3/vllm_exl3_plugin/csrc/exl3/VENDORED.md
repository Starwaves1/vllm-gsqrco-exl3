# Vendored exllamav3 (EXL3 dense linear)

Source: https://github.com/turboderp-org/exllamav3 tag v1.5.3, commit
d3739fd393337b1ff4d6c2a342b12f0c87a9592f (master, 2026-09-27), directory
`exllamav3/exllamav3_ext/`. License: MIT, see `LICENSE` (copied verbatim from the
repository root at the same commit).

Files are byte-identical copies at their upstream paths relative to `exllamav3_ext/`.
**Do not edit them**; all adaptation lives in `../exl3_shim.cu`. To update, re-copy the
same list from a new commit, recompute the hashes below and rebuild. Check: `cmp` each file
against a checkout (`git clone https://github.com/turboderp-org/exllamav3 && git checkout
d3739fd`).

What is here: the `#include "..."` closure of the dense-linear entry points and every
`.cu` they need to link: `quant/{exl3_gemm, exl3_gemv, exl3_gemv_int8, exl3_kernel_map,
exl3_devctx, coop_autotune, hadamard, reconstruct, frac}.cu`, `hgemm.cu`,
`hgemm_f16acc.cu`, `graph.cu`, and the kernel instances in `quant/comp_units/`
(`exl3_comp_unit_*` for K 1..8 x codebook {3inst, mcg, mul1} and half-integer K 1.5/2.5/3.5,
`exl3_gemv_half_inst`, `exl3_gemv_int8_inst_*`). 94 files, 60 of them compiled, 381 KB.

Not vendored: attention, GDN, sampling, MoE (`exl3_moe*`), `bindings.cpp`, `libtorch/`
and `cuda_drv.cpp`. `graph.cu` (exllamav3's own CUDA graph recorder) calls
`CudaDrv::instance()` from `cuda_drv.cpp`; the shim defines a stub that throws, because
the shim never passes a `Graph` (so the recorder never runs and libcuda stays out of the
link). `graph.cu` is compiled only because `exl3_gemm.cu` references `Graph` members.

Compiled (`plugin-exl3/setup.py`, `VLLM_EXL3_BUILD=1`): all 60 `.cu` below plus
`../exl3_shim.cu`, with exllamav3's nvcc flags (`-O3 --use_fast_math`, diag 177/20012
suppressed; `--compress-mode` left out).

Files (sha256):

```
763a7389e87cca199e9a2eb1d329e12f2acaee423f7a301e6575b694f8f13090  ./arch.cuh
dd6038fa6eb8b28df56b184e358e5633421af76eeb40bad4abc522ce6ba54f57  ./compat.cuh
0d2efdd0a3a9271d7d2cbbcc813343f42aa169746eeb56b9fbd707569581ae0b  ./cuda_drv.h
8e57b2f599df4df9ac7cdd06dd7e175d511e5780b59e9b6f3570c3476f0a7901  ./graph.cu
6fe8e3d3ab39d885ff8447f6f8f3397c35387619722fca0222eddcc753b6d074  ./graph.cuh
f763c6a182a71352ceda5cee35a1f3b3b8b2bbb5a493886f32f4070d1b02bf7a  ./hgemm.cu
d190af31f3139cffa1f9de35f63484a20e539c44b815eb8676000fcab2c41731  ./hgemm.cuh
c4e202d51d57d4108c692f5688b97ab5c68455c0896a8150e1df36091e2e6c31  ./hgemm_f16acc.cu
27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff  ./LICENSE
266eabb4e1e5cded91dcc5e7f68293cefd68df84e9deb079dbfd6b04884bab2f  ./ptx.cuh
14f8c19d5edf07244c2b337f6d12386e02b811d6473a57ff116529049782e10d  ./quant/bits_k.cuh
0e3c63b323f8d3cc15c6a8f2e2b3816efafd71e149a99997b30b2f806375138a  ./quant/codebook.cuh
83215843cbf76b36f4571fee609b8448770554f6173ebe027c9d90de146fe853  ./quant/comp_units/exl3_comp_unit_1_cb0.cu
56e67843999745a63ea87f48bbad8146bcd755117e5121d87f514953586d345f  ./quant/comp_units/exl3_comp_unit_1_cb1.cu
2a3646ad363332cff429ddd174de356ca28500f840c9e0b7d5ab2c75c50d2204  ./quant/comp_units/exl3_comp_unit_1_cb2.cu
ec583257b5a07d10a0d4fa137822009e89de977190ffce4cb1ebd05296929590  ./quant/comp_units/exl3_comp_unit_1.cuh
c722f5fe9d0d8359eff035c194cf4144b70e90663a38b585eef9fa6dcebe1911  ./quant/comp_units/exl3_comp_unit_2_cb0.cu
acede1fcdfa55d40b3616053e3a6f3cc903b6f0e2e5e4508512315fe8494ad7d  ./quant/comp_units/exl3_comp_unit_2_cb1.cu
4d3f082b9b7c744bcb6f86a5ac5213dd9b1d74ffb86aa47558cadcb8dddb418a  ./quant/comp_units/exl3_comp_unit_2_cb2.cu
bc5a914a36350dcbde26024dd221aa97d955ef67150fcf45bf9833cd81c5c68f  ./quant/comp_units/exl3_comp_unit_2.cuh
8bde236973f6bb9f6884488bab1be6cfe4e1e9ea8b7a82ef7281fe3542c70b2a  ./quant/comp_units/exl3_comp_unit_3_cb0.cu
a36ceebc1eafe329c6441585b663bc8d57bba61a3e0208490c63dc0a6ebb7ea0  ./quant/comp_units/exl3_comp_unit_3_cb1.cu
aed62c260b031e36d40f7c21f02afaf22a1333a0a61f8ea9efd793331f2030c6  ./quant/comp_units/exl3_comp_unit_3_cb2.cu
eacc6ee8968efde79e1830109fac21d1c571d1818d3cff714169104731d231cf  ./quant/comp_units/exl3_comp_unit_3.cuh
6a5d54ff1c4b2943670988852781d4f80b94d78eab03ebd6da7e8c725dfed16e  ./quant/comp_units/exl3_comp_unit_4_cb0.cu
d7b2bd80d427c6f0d57503d6b203c8fba400f5cf8a0cae1587dff5c0d364bcb3  ./quant/comp_units/exl3_comp_unit_4_cb1.cu
c937daea97703a2c91d75262aba886d90fb36f6cf66645cb576f7f47e662a3ea  ./quant/comp_units/exl3_comp_unit_4_cb2.cu
313d479451fe5ecbbe38fb2952fbca4a8526cfbbed2363e10c86a47390b40c84  ./quant/comp_units/exl3_comp_unit_4.cuh
d4f9cb01b667b36fa8f8a0fcc805f26dc20ab44e77ec12cbcf5cfd841217490f  ./quant/comp_units/exl3_comp_unit_5_cb0.cu
07c53ef89d7c12a4af53ce24b4dd805c3e2f701a08fc665db1f6d8188ecc2821  ./quant/comp_units/exl3_comp_unit_5_cb1.cu
763dfed400ecd84a77a0f6f9331a903b7f5cc00272418ddb20cadf397493541d  ./quant/comp_units/exl3_comp_unit_5_cb2.cu
f377ef4dd47edca548605a4f48940b11a1ff93b0402855f24a1a4cc20e6e3059  ./quant/comp_units/exl3_comp_unit_5.cuh
52550c9af7528cc7d04b89478a3d74bbc75099e5fbfffa8537f86d81a639299b  ./quant/comp_units/exl3_comp_unit_6_cb0.cu
564e1a8847568bdfd4cfbf0077f63fa1d79f598c9bac3fd1b7fb4401bdcbd622  ./quant/comp_units/exl3_comp_unit_6_cb1.cu
9db09ffaefa95fc17110b4fcd8dbc407b37d872ce2142f79f8b5b1c97fb288de  ./quant/comp_units/exl3_comp_unit_6_cb2.cu
11497c8618830b9e5849eac8ac516585f69c8763620a23214a6fa1340cc0d145  ./quant/comp_units/exl3_comp_unit_6.cuh
55dad76604e7c4755effd6749135dd2abbbfe1521c3508330e64e5b8aafb3e5f  ./quant/comp_units/exl3_comp_unit_7_cb0.cu
655960437b2611af94020ab73b6300a068f1c8dfa0ff05b05ea57d1f8a4387e0  ./quant/comp_units/exl3_comp_unit_7_cb1.cu
2ea0a1e4e245cf18963fce6decdbfa658fae1a28f671c375483d968f474c02c7  ./quant/comp_units/exl3_comp_unit_7_cb2.cu
47b24948304d803fee460f0811b362e8d9f2b311c1d53f65aab493e96adb2be2  ./quant/comp_units/exl3_comp_unit_7.cuh
d4cccf2a87c8951a0da1fb2b0d2fa41c67d9e656ba14f91fec5154d72bf6c3a9  ./quant/comp_units/exl3_comp_unit_8_cb0.cu
44497b3d7f670d9438a457203924d0298b0aa2db3060d086b85ec6af547fef25  ./quant/comp_units/exl3_comp_unit_8_cb1.cu
768046fb7d290b10d804c98863eef5cc35ac621b661211553f266ca4b5c3c303  ./quant/comp_units/exl3_comp_unit_8_cb2.cu
3354be2f6db645bf48c6244c7ae397716be66bad121f7c1e9b39d265d7da56ea  ./quant/comp_units/exl3_comp_unit_8.cuh
45852e21e1f63a3d4e457ab1bf345dc23d2b76ebb392a94e40bc0af9671e447f  ./quant/comp_units/exl3_comp_unit_h1.cu
3a00561b09fa5846cf5ad6af833cfe79b461818a16b98da1ce7116d5ef543496  ./quant/comp_units/exl3_comp_unit_h2.cu
9de9d155f0c180fd6058e676865f43162d8c1a07778a19845757e0ff6c2cc17e  ./quant/comp_units/exl3_comp_unit_h3.cu
3e4b7ef3ba8ab32c0d751da372c4016dd14fd2c1e937b609d37d6ec1e16e52ed  ./quant/comp_units/exl3_gemv_half_inst.cu
de0d815c65769541dcfe0854ca7903c6cd44d05718f6e550ef6bdc92fc155861  ./quant/comp_units/exl3_gemv_int8_instances.cuh
4a5c0a6c5b2d4796de23f039c25335985054fd6d4f22aae2c34caf001270e61a  ./quant/comp_units/exl3_gemv_int8_inst_coop_h1.cu
d5eadd38d421d00b4d2ea2e22d5a705d70e647668c78f98a5e3d4ffe8734a6c7  ./quant/comp_units/exl3_gemv_int8_inst_coop_h2.cu
6596ea626958c3e68953f22dc1c8e8d33d1e35620bf9da94fa7d73420e222728  ./quant/comp_units/exl3_gemv_int8_inst_coop_h3.cu
58d936f3aab355821bc3a8becebf5f9416cb5bebc65d3019b7b9f809bbe42df8  ./quant/comp_units/exl3_gemv_int8_inst_coop_k1.cu
efd08cbabd409eec5cd747f7a2b8aec6dd2e32d0cf27620724cd35c51796c233  ./quant/comp_units/exl3_gemv_int8_inst_coop_k2.cu
02a7686644e617dd669ed3580884530828a766d6244d4a8fd08d6512e9059b9f  ./quant/comp_units/exl3_gemv_int8_inst_coop_k3.cu
0607610b849147c644d8131a756702927942dada1dd8b1dc205be84705d819d2  ./quant/comp_units/exl3_gemv_int8_inst_coop_k4.cu
ffec88863674e28d93dcc5408f4424c201d5d56d01375060b2f5b18f01b29c59  ./quant/comp_units/exl3_gemv_int8_inst_coop_k5.cu
2db74af9d7d96d730b4baffd8595725430e6b464705ea7fbaf2416db3089a7a6  ./quant/comp_units/exl3_gemv_int8_inst_coop_k6.cu
201d3074990c89c843c44c74c518a15d38da35d6940d1db5f91fa1ee289cd455  ./quant/comp_units/exl3_gemv_int8_inst_coop_k7.cu
7dd28f90ca4c6e0e4044efda016643d6550c2aa59663df5149bcde5b92787bd7  ./quant/comp_units/exl3_gemv_int8_inst_coop_k8.cu
3a74b8e3bb49f76932d5845ae3fb2d3baf67babedbefc06691b737e41216bf14  ./quant/comp_units/exl3_gemv_int8_inst_sq_h1.cu
e307b8fb05f2346b54acc0f03ca34003e423b020d55a30d5b044d3b445ba72c7  ./quant/comp_units/exl3_gemv_int8_inst_sq_h2.cu
ab42f217151f76ccea5689fc15ea757c6a95244fb28bd69a4c4adc0ffa483624  ./quant/comp_units/exl3_gemv_int8_inst_sq_h3.cu
4445110177350ad011f5e40556031a32a7103f609771c2debc53029fe08bdf4f  ./quant/comp_units/exl3_gemv_int8_inst_sq_k1.cu
62cda3157691f01faf9a74628aa903bfa8ffbcf9932f98115a8e8c1a41642e83  ./quant/comp_units/exl3_gemv_int8_inst_sq_k2.cu
c2db08bbd4bf49c426d0f8ca38c3bc1c6fa178586e3785f23a1a5d65840f8c52  ./quant/comp_units/exl3_gemv_int8_inst_sq_k3.cu
ef3abf2ac86e607f022009bb6854b23ca9303ca03c0f978840d432ad4dc52f60  ./quant/comp_units/exl3_gemv_int8_inst_sq_k4.cu
35e1917688dcbaa5b2ef752bdd17ecddb71a7c3b29066956c0e21572a7b292a1  ./quant/comp_units/exl3_gemv_int8_inst_sq_k5.cu
c4fdeadd2db56d64872168750bb18639c414def84a4e529ee8b28bcae1d0c1e1  ./quant/comp_units/exl3_gemv_int8_inst_sq_k6.cu
b685a7e18b470b86b68dfdddedc9d419b40e21462cb31c7738f44c105e86888c  ./quant/coop_autotune.cu
1a2bfaae6151ea45229d839885099803a028896b8ad70fc322e832a4b5213b71  ./quant/coop_autotune.cuh
7115effb88a65468297015ced4f2ad50156502e859f8d207d16fb7fb413c882c  ./quant/exl3_devctx.cu
7f3d88fcbac664e4a00ec0374332056d3bd24355951a78e99ec34a09a739ed93  ./quant/exl3_devctx.cuh
4e48a37af4811e8e0f7e00e86c29c9c044fa0b6a8f6a685855a033684e4d6552  ./quant/exl3_dq.cuh
d13d924b7deba592f8d57f9ee6415d3eaefcd4177dff04d83659883c87e31cf4  ./quant/exl3_gemm.cu
606ec462784d2550707df83e052447d111ad9d59ae64806dd272ef2f898a0b9f  ./quant/exl3_gemm.cuh
3aa69924e7e32e0b442798cd0b82e08b82af4557bad37f7425bd0d47f7701364  ./quant/exl3_gemm_inner.cuh
3e94f9e1f3acb0dd1ac2d66b16e767fd993f798adcd0c957969ae9adc1aa8809  ./quant/exl3_gemm_kernel.cuh
2e69e32a6718ffae07986545c4bffcf0fdf42973b59654c3b34c3fc55f714a86  ./quant/exl3_gemv.cu
46c6ee7c303b4c3727363aa64ece118b8be9fd169a94e1f505a37b77f1702d1c  ./quant/exl3_gemv.cuh
9234a6ec7d3777b33e8511797ad3a0201facccbeabb3a496f6231bbc24f7da79  ./quant/exl3_gemv_int8.cu
2604b78a64537e072913dfe3ffb31e95f36fe2fce27943890d3e91fd17a5e636  ./quant/exl3_gemv_int8.cuh
c90bb879f2bb2e085eb0a32659ece27160118651742df67b8f5ea29418e19baa  ./quant/exl3_gemv_int8_kernel.cuh
c10d0bbd5e1f62506ec95ac35f07f55edf3b4773606bdcc8ee1b82ceacf37070  ./quant/exl3_gemv_kernel.cuh
d0ae0768f0f02ad6fb2f0da4634701ce3d8d88375120d5ee009bf758339e6fa8  ./quant/exl3_kernel_map.cu
7c65714afa2abd17d25d75571bdab6576067635f3ce619d9d3212853700dab58  ./quant/exl3_kernel_map.cuh
d4e9941a7e87abfb1e587466cbd81b663c1808f64fad336a825bcdd41d3508b7  ./quant/frac.cu
cb5ba137de7bf3cee25b1eea2eac909e90e0f5567c0e2467215984bcc19e722b  ./quant/frac.cuh
7df512a2d5c3c2cee7da707d7776f661f8f19b84f173b668612cc842fed3652f  ./quant/hadamard.cu
8d8e437aced88735e919563301ffac0e4a2aac28cc542ed3f738e1216ea0c36b  ./quant/hadamard_inner.cuh
3c2b5ac4b1e718dcfaf8f8e974d4b8a19f1d4d1be54068772bf2b9901a340ee2  ./quant/quantize.cuh
3541a9e4bfa2fec9d2c9d420ee5d003946962ac77e7a5fa20a07cb012026e548  ./quant/reconstruct.cu
4c9049c044ab1b1ca1cc2610dc06d9e599c0507248909939c12d96024314d53b  ./quant/reconstruct.cuh
1907cea115260db7c3b0de540e375e7733b7ffc9019bec190c1d92abea1d964e  ./util.cuh
ba89ac6793bf31cfe123d13baa2e34531a18a24bc3b2ff4add8cf4856b75a1c8  ./util.h
```
