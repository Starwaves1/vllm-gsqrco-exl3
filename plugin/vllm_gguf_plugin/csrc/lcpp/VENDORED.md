# Vendored llama.cpp ggml-cuda (Route L)

Source: https://github.com/ggml-org/llama.cpp tag b11211, commit
d7fb90e8e2494b2908934d956a3202fd60152ee0 (local checkout ~/llama.cpp-b11211).
License: MIT, see `LICENSE` (copied verbatim from the same commit).

Files are byte-identical copies at their upstream relative paths. **Do not edit
them**; all adaptation lives in `../lcpp_shim.cu`, owned kernels in
`../lcpp_shim.cu` and `../lcpp_owned_*.cu`. To update, re-copy the same
list from a new commit and rebuild. Check: `cmp` each file against the checkout.

Only what the MMVQ / MMQ / q8_1-quantize path needs (found with `nvcc -M`), and
MMQ instances for the 9 MMQ-capable types in Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp
(IQ1_M has no MMQ upstream; it stays on the plugin's own kernels). The
`mmq-config-*.cuh` for other GPUs are included unconditionally by `mmq.cuh`.

`mmq.cu` is deliberately not vendored: its type switch references all 22 MMQ
instances and it needs `mmid.cu`. The shim instead mirrors its non-MoE q8_1
branch (`mul_mat_q` in `../lcpp_shim.cu`) and calls `mul_mat_q_case<type>` from
the 9 instances directly.

Compiled (setup.py, `VLLM_GGUF_BUILD_LCPP=1`): `mmvq.cu quantize.cu` and the 9
`template-instances/mmq-instance-*.cu`.

Files (sha256):

```
94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23  ./ggml/include/ggml-alloc.h
46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2  ./ggml/include/ggml-backend.h
9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042  ./ggml/include/ggml-cuda.h
12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9  ./ggml/include/ggml.h
e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4  ./ggml/include/gguf.h
0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c  ./ggml/src/ggml-common.h
6010078d639db19b1e8a2f78c4661ad240659b7bfa5a22864e315a2cbbc8da9b  ./ggml/src/ggml-cuda/common.cuh
08a18ccf4400990e9b2a651fdf702f49f0a521e427f151443e4359c912ac4fab  ./ggml/src/ggml-cuda/mma.cuh
6de9336a6c2f65c0ad83cbe9303dcd03e8cbb6185d15346133a31421000f6408  ./ggml/src/ggml-cuda/mmq-config-ampere.cuh
8bec1e65a3a2a58fab1f406225f6ba2f1a4c38a19793dfa255baff69bc044d70  ./ggml/src/ggml-cuda/mmq-config-blackwell.cuh
314eeb8453834723d5224012f692f70673f86ecc42d2b6d6f2005a0fad124dbc  ./ggml/src/ggml-cuda/mmq-config-cdna.cuh
f669a723fc19d1e650de33a2852a10472a5773b0c8a3cd5b96662bc4fe6ebfe3  ./ggml/src/ggml-cuda/mmq-config-gcn.cuh
67a93b6d26d3e16e77fd285dd77e7cca7848fb0b0d68897c2ff1e8cc47102263  ./ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh
48bae44f1f0e667fa02fb898859eb18c03961bdbadf53c1e22d98801ed70affe  ./ggml/src/ggml-cuda/mmq-config-pascal-older.cuh
758be2a33b89018841f432c15cd63009ad212d71eae888a04b4535cb64941aa7  ./ggml/src/ggml-cuda/mmq-config-rdna2.cuh
5243cb07aded29b0bd3254cf9f9856989f53b133a73874388282a35c3990f0cd  ./ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh
a1ac4f4fa799f2fb200ead23f3403bb0f729d41109b8967c6d76216768aae0cf  ./ggml/src/ggml-cuda/mmq-config-rdna3.cuh
e40c9c5fe85607db07ae351dd6ca13c92caaa238e2a62d5134877d9fbcb5c645  ./ggml/src/ggml-cuda/mmq-config-rdna4.cuh
b4337f6777bde524b1e8af4aabea6edb5d58f1fa4d6986f4b50cc7e2f05290f0  ./ggml/src/ggml-cuda/mmq.cuh
83428f3c7914b6d8103660a8cd2f4bc59e93c30c29fb2878a5f3b8577668fe1e  ./ggml/src/ggml-cuda/mmq-load-tiles.cuh
08339a456dda4ff05612ec197ccab7010bef9e5904b5ac812363615009613a22  ./ggml/src/ggml-cuda/mmq-vec-dot.cuh
df8ddb22f31acb058d1819fcebcc1c280c986dd0fee8a88b6c72514a16e1f5ed  ./ggml/src/ggml-cuda/mmvq.cu
b37f8359f519ba90ddecd9cb3ca5f8c2ae05685bfae6813d435d75763c2bb006  ./ggml/src/ggml-cuda/mmvq.cuh
71cd7baab62ee1f457d23ae648c381d18b9b1327ec34832f31463b14b81895b3  ./ggml/src/ggml-cuda/quantize.cu
112e190a096e1fa68d4c72ca2a450cb202586e3f2919c884f3e3316a8d5c57cd  ./ggml/src/ggml-cuda/quantize.cuh
8a841418b1b818a13f739759ea0d36ed778e1b6343aee85fdc58d341b169bd55  ./ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu
549d0c56b4b7ffccf262a611e28d257ce8e7ee94ce7ac814259c427eaeb10fe8  ./ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu
7fa1a4a0aaec59e2e27eb40a13078be685952e446c5c9fb0840a47d0e28a17bf  ./ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu
8167508364494a8f51f736186c5294585a26302cb034e051b8b80acf1db18850  ./ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu
13af5ff17c13dbbe712a6f400b83b69da77792969162c6f345491392d36c1d51  ./ggml/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu
c4f568105a3ff273287b04c58879ea455efd5fa05aa72da9ef0efeb7ab9de520  ./ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu
e378922ee4aaabe395e7e3f620a38ae26e36e27b46036b8b7fea71292bf8cc2b  ./ggml/src/ggml-cuda/template-instances/mmq-instance-q2_k.cu
2ab47361d40a9ece25406c20693424a466c93bcb5cdb5c06cee8f9d03a806beb  ./ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu
989943d6501a74927d079eacca793e94a1233863196c94dd20e459017ed76fc2  ./ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu
a6903bb02577a33b79c738e62e425867f9122d1b4981a3b228062b4f978b2858  ./ggml/src/ggml-cuda/unary.cuh
a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe  ./ggml/src/ggml-cuda/vecdotq.cuh
e7847065d5d741f892d2f1f7aaaa8f418a41e3b983d1e6f09010055d4d1c38d1  ./ggml/src/ggml-cuda/vendors/cuda.h
43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3  ./ggml/src/ggml-impl.h
94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d  ./LICENSE
```
