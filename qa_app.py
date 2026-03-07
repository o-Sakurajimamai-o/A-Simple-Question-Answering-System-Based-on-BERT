import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import torch
import gradio as gr
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

# ==================== 模型加载 ====================

BERT_PATH = "./best_bert_model"
T5_PATH = "./best_t5_model_2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"正在加载模型，设备: {device}")

print("加载 T5 纠错模型...")
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH).to(device)
t5_model.eval()

print("加载 BERT 问答模型...")
bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
bert_model = BertForQuestionAnswering.from_pretrained(BERT_PATH).to(device)
bert_model.eval()

print("模型加载完成！")


# ==================== 推理函数 ====================

def correct_text(text: str) -> str:
    """使用 T5 模型对输入文本进行纠错"""
    input_text = f"纠错：{text}"
    inputs = t5_tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = t5_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    corrected = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected.replace(" ", "")


def bert_qa(context: str, question: str) -> str:
    """使用 BERT 模型从原文中抽取答案"""
    inputs = bert_tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offset_mapping = inputs.pop("offset_mapping")
    inputs.pop("overflow_to_sample_mapping")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    best_score = float("-inf")
    best_answer = ""

    for i in range(inputs["input_ids"].shape[0]):
        start_logits = outputs.start_logits[i]
        end_logits = outputs.end_logits[i]
        offsets = offset_mapping[i]
        input_ids = inputs["input_ids"][i]

        sequence_ids = bert_tokenizer(
            question, context,
            truncation="only_second",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        ).sequence_ids(i)

        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        for s in range(context_start, context_end + 1):
            for e in range(s, min(s + 50, context_end + 1)):
                score = start_logits[s].item() + end_logits[e].item()
                if score > best_score:
                    best_score = score
                    start_char = offsets[s][0].item()
                    end_char = offsets[e][1].item()
                    best_answer = context[start_char:end_char]

    return best_answer if best_answer else "（未找到答案）"


def predict(context: str, question: str):
    """完整推理流程：T5 纠错 → BERT 问答"""
    if not context.strip():
        return "请输入原文", "", ""
    if not question.strip():
        return "请输入问题", "", ""

    corrected_question = correct_text(question)

    answer = bert_qa(context, corrected_question)

    if corrected_question == question:
        correction_display = "（无需纠正）"
    else:
        diff_chars = []
        for i, (orig, corr) in enumerate(
            zip(question, corrected_question[: len(question)])
        ):
            if orig != corr:
                diff_chars.append(f"「{orig}」→「{corr}」")
        if len(corrected_question) != len(question):
            diff_chars.append(f"（长度变化: {len(question)}→{len(corrected_question)}）")
        correction_display = "  ".join(diff_chars) if diff_chars else "（细微调整）"

    return corrected_question, correction_display, answer


# ==================== Gradio 界面 ====================

with gr.Blocks(
    title="双模型中文QA系统",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 双模型中文 QA 系统
        **T5 纠错** + **BERT 阅读理解** 联合推理
        输入一段原文和一个问题（可以故意打错字），系统会先纠正问题，再从原文中找出答案。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            context_input = gr.Textbox(
                label="原文 (Context)",
                placeholder="请输入一段文章...",
                lines=8,
            )
            question_input = gr.Textbox(
                label="问题 (Question)",
                placeholder="请输入问题（可以包含错别字）...",
                lines=2,
            )
            submit_btn = gr.Button("提交", variant="primary")
            clear_btn = gr.ClearButton(
                [context_input, question_input], value="清空"
            )

        with gr.Column(scale=1):
            corrected_output = gr.Textbox(label="纠正后的问题", interactive=False)
            diff_output = gr.Textbox(label="纠错详情", interactive=False)
            answer_output = gr.Textbox(label="答案", interactive=False)

    submit_btn.click(
        fn=predict,
        inputs=[context_input, question_input],
        outputs=[corrected_output, diff_output, answer_output],
    )

    question_input.submit(
        fn=predict,
        inputs=[context_input, question_input],
        outputs=[corrected_output, diff_output, answer_output],
    )

    gr.Examples(
        examples=[
            [
                "故宫又称紫禁城，位于北京市中心，是明清两代的皇家宫殿。"
                "故宫占地面积约72万平方米，建筑面积约15万平方米，"
                "有大小宫殿七十多座，房屋九千余间。"
                "故宫于明朝永乐十八年（1420年）建成，至今已有六百多年的历史。",
                "故宫是什么时侯建成的？",
            ],
            [
                "华为技术有限公司成立于1987年，总部位于广东省深圳市。"
                "华为是全球领先的信息与通信技术解决方案供应商，"
                "业务遍及170多个国家和地区，服务全球三分之一以上的人口。",
                "华为的总布在哪里？",
            ],
        ],
        inputs=[context_input, question_input],
    )


if __name__ == "__main__":
    demo.launch()
