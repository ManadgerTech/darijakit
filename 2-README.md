# DarijaKit 🦁

**A developer toolkit for Moroccan Darija (الدارجة المغربية) on local LLMs.**

Arabizi transliteration, Darija/Arabic normalization, and a benchmark suite
that measures how well any local model actually understands Darija — all
running on your own machine, against Ollama, llama.cpp, LM Studio or vLLM.

Made in Morocco by [Manadger Tech](https://manadgerhost.com).

---

## Why

Hundreds of millions of people speak Arabic dialects that LLMs handle
poorly, and Moroccan Darija is among the least served. People write it in
Arabic script *and* in Arabizi (`ch7al mn sa3a khasni nkhdem?`), mix it
with French, and spell the same word five different ways. Existing work is
either a research benchmark or a closed product. DarijaKit is the missing
practical toolkit:

- **Normalize** Darija text before it hits your model (spelling variants,
  diacritics, alef/ya forms).
- **Transliterate** Arabizi to Arabic script so models trained on Arabic
  script can read it.
- **Benchmark** any local model on real Darija tasks — comprehension,
  translation, sentiment, entities, grammar, culture — and get a Markdown
  or JSON report.
- **Fine-tune** a small open model on Darija with the included LoRA recipe,
  sized for a single 12 GB GPU.

Zero runtime dependencies. Python 3.9+. MIT licensed.

## Install

```bash
pip install git+https://github.com/ManadgerTech/darijakit.git
```

Or just copy `darijakit.py` into your project — it is a single stdlib-only
module.

## Quickstart

```bash
# Transliterate Arabizi to Arabic script
darijakit arabizi "ch7al mn sa3a khasni nkhdem?"
# → شحال من ساعة خاصني نخدم?

# Normalize Darija spelling variants
darijakit normalize "بش نمشي للسوق داب"
# → باش نمشي للسوق دابا

# Detect what script a message is written in
darijakit detect "salam labas 3lik?"
# → arabizi

# Benchmark a local model on the Darija eval set (Ollama running)
darijakit eval --model qwen2.5:7b
```

As a library:

```python
import darijakit as dk

dk.arabizi_to_arabic("bghit nakol tajine")     # → "بغيت ناكل طاجين"
dk.normalize("شكراً بزاف")                        # → "شكرا بزاف"
dk.detect_script("fin ghadi?")                  # → "arabizi"

items = dk.load_evalset("evalset.jsonl")
report = dk.run_eval(items, model="qwen2.5:7b")  # needs a local endpoint
print(dk.report_to_markdown(report))
```

## The benchmark

`evalset.jsonl` ships with 30 hand-written Darija items across six
categories:

| Category | What it measures |
| --- | --- |
| comprehension | Does the model understand Darija words and idioms? |
| translate_to_msa | Darija → Modern Standard Arabic |
| translate_to_english | Darija → English |
| sentiment | Sentiment in Darija text |
| entities | Names, numbers and places in Darija sentences |
| grammar / culture | Darija grammar and Moroccan cultural knowledge |

Scoring is deterministic (option letters, normalized containment), so
results are comparable across models and over time. Bring your own items by
adding JSONL lines: `{"id", "kind", "prompt", "expected", "category"}`.

Works with any OpenAI-compatible endpoint:

```bash
# Ollama (default)
darijakit eval --model qwen2.5:7b

# llama.cpp server
darijakit eval --model local --endpoint http://localhost:8080/v1

# JSON for CI
darijakit eval --model qwen2.5:7b --json > results/$(date +%F).json
```

## Fine-tuning

`finetune_lora.py` is a ready LoRA recipe: 4-bit quantized base + LoRA
adapters, sized for one 12 GB GPU (RTX 3500 Ada / 3060 12GB class):

```bash
pip install -r requirements-finetune.txt
python finetune_lora.py --dataset sample_instruct.jsonl --output ./darija-lora
```

`sample_instruct.jsonl` is a small seed set of Darija instruction pairs to
show the format. Grow it with your own data — the eval set tells you
whether the fine-tune actually helped.

## Tests

```bash
python -m unittest tests_darijakit -v
```

## Roadmap

- Larger community-driven eval set (contributions welcome)
- Darija ↔ French code-switch handling
- Ollama Modelfile template for a fine-tuned Darija model
- Hugging Face dataset release of the eval set

## Contributing

Issues and pull requests are welcome — especially new eval items and
spelling-variant mappings from native speakers. If you speak Darija, you
can contribute.

## License

MIT © Manadger Tech SARL

---

## بالدارجة / بالعربية 🇲🇦

**DarijaKit** هو صندوق أدوات للمطورين باش يخدمو الدارجة المغربية مع النماذج
اللغوية المحلية (Ollama، llama.cpp...):

- **تحويل الأرابيزي** (`ch7al` ← `شحال`) للحروف العربية.
- **توحيد الكتابة**: كل واحد كيكتب الدارجة بطريقة، الأداة كتوحدها.
- **تقييم النماذج**: 30 سؤال بالدارجة (فهم، ترجمة، مشاعر، ثقافة) باش تعرف
  واش النموذج ديالك كيفهم الدارجة بصح ولا لا.
- **التدريب**: وصفة LoRA جاهزة باش تدرب نموذج صغير على الدارجة حتى فجهاز
  بكارت شاشة 12GB.

كلشي كيخدم محليا، بلا ما تسيفط والو. الكود مفتوح المصدر (MIT) ومرحبا
بالمساهمات، خصوصا من اللي كيهضرو الدارجة.

```bash
darijakit eval --model qwen2.5:7b
```
