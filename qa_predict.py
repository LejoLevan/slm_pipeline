from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from peft import PeftModel

# Context:
# Infosys is a global leader in technology services and consulting. 
# The InStep Internship Program is an award-winning initiative offering real-world projects. 
# It spans domains like Artificial Intelligence, Full Stack Development, Business Analysis, Data Science, Cybersecurity. 
# Interns receive benefits such as mentorship, access to training resources, opportunities to contribute meaningfully to projects. 
# On May 19th, 2025, the Hartford branch welcomed 15 interns: 4 in Business Analysis and 11 in Technology. Of the Technology interns, 9 worked on Full Stack Development 
# and 2 focused on Artificial Intelligence. Each intern was paired with a mentor, such as Manas Ranjan Kar.

# Questions:
# On what date did the internship program start at the Hartford branch?
# How many interns joined in Hartford?
# What was the number of tech interns at the Hartford branch?

# Impossible Question:
# Which universities did the interns graduate from?

def main():
    test = True

    base_model_path = "models/bert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    if test:
        print("Using Pipeline Trained Model (Test)")
        model = PeftModel.from_pretrained(model, "output/bert-base-uncased")
    else:
        print("Using Base Model (Control)")

    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    context = "Infosys is a global leader in technology services and consulting. The InStep Internship Program is an award-winning initiative offering real-world projects. It spans domains like Artificial Intelligence, Full Stack Development, Business Analysis, Data Science, Cybersecurity. Interns receive benefits such as mentorship, access to training resources, opportunities to contribute meaningfully to projects. On May 19th, 2025, the Hartford branch welcomed 15 interns: 4 in Business Analysis and 11 in Technology. Of the Technology interns, 9 worked on Full Stack Development and 2 focused on Artificial Intelligence. Each intern was paired with a mentor, such as Manas Ranjan Kar."
    while(True):
        question = input("Enter Question: ")
        if question == "":
            break
        result = qa_pipeline(
            context = context,
            question = question
        )
        print(result)

if __name__ == "__main__":
    main()