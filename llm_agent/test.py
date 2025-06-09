from model import build_deepseek_grad_engine
import textgrad as tg

engine = build_deepseek_grad_engine()
tg.set_backward_engine(engine, override=True)

model = tg.BlackboxLLM(engine)
question_string = ("If it takes 1 hour to dry 25 shirts under the sun,how long will it take to dry 30 shirts under the sun? Reason step by step")
question = tg.Variable(question_string, role_description="question to the LLM", requires_grad=False)
answer = model(question)
print("first answer:", answer)
print('-'*50)
answer.set_role_description("concise and accurate answer to the question")

optimizer = tg.TGD(parameters=[answer])
evaluation_instruction = (f"Here's a question: {question_string}. "
"Evaluate any given answer to this question, "
"be very smart, logical, careful, and critical. "
"Just identify the error in the answer and provide concise feedback.")

loss_fn = tg.TextLoss(evaluation_instruction)
epoch = 5
for _ in range(epoch):
    loss = loss_fn(answer)
    loss.backward()
    optimizer.step()
    print(answer.value)
    print('-'*50)

print('final answer:', answer.value)
