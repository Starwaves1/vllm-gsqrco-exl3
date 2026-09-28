"""Which GGUF adapter does the MTP drafter get, and what does its model allocate? (no weights loaded)"""
import sys, json
sys.argv = ["x"] + [l.rstrip("\n") for l in open("/tmp/p1diag/argv.txt")]
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
import vllm.plugins; vllm.plugins.load_general_plugins()
p = make_arg_parser(FlexibleArgumentParser())
a = p.parse_args(sys.argv[1:])
ea = AsyncEngineArgs.from_cli_args(a)
cfg = ea.create_engine_config()
sc = cfg.speculative_config
dmc = sc.draft_model_config
print("target model_type", cfg.model_config.hf_config.model_type, cfg.model_config.architectures)
print("draft model_type", dmc.hf_config.model_type, dmc.architectures, "quant", dmc.quantization)
from vllm_gguf_plugin.weights_adapter import get_weights_adapter
print("draft adapter", type(get_weights_adapter(dmc.hf_config)).__name__)
print("target adapter", type(get_weights_adapter(cfg.model_config.hf_config)).__name__)
tc = dmc.hf_config.get_text_config()
print("vocab", tc.vocab_size, "hidden", tc.hidden_size, "tie", getattr(tc, "tie_word_embeddings", None))
