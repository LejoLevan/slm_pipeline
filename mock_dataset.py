import uuid
import random
import json
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
Faker.seed(42)

base_info = {
    "company": "Infosys",
    "program_name": "InStep Internship Program",
    "domains": [
        "Artificial Intelligence",
        "Full Stack Development",
        "Business Analysis",
        "Data Science",
        "Cybersecurity"
    ],
    "benefits": [
        "mentorship",
        "access to training resources",
        "opportunities to contribute meaningfully to projects"
    ]
}

# Question templates with variations for each type
question_templates = [
    ([
        "On what date did the internship program start at the {city} branch?",
        "What is the exact date when interns joined the {city} branch?",
        "When did the internship program kick off in {city}?",
        "Can you tell me the start date of the internship in {city}?"
    ], "date"),
    ([
        "How many interns joined in {city}?",
        "What was the total number of interns at the {city} branch?",
        "How large was the intern cohort in {city}?",
        "Number of interns that started in {city}?"
    ], "total_interns"),
    ([
        "How many interns worked in Business Analysis in {city}?",
        "Business Analyst intern count in {city}?",
        "Number of Business Analysis interns in {city}?",
        "What was the BA intern headcount at the {city} branch?"
    ], "ba_interns"),
    ([
        "How many technology interns were there in {city}?",
        "What was the number of tech interns at the {city} branch?",
        "Tech intern count in {city}?",
        "Number of interns in the technology track at {city}?"
    ], "tech_interns"),
    ([
        "How many tech interns worked on Artificial Intelligence in {city}?",
        "Number of AI interns at the {city} branch?",
        "AI-focused intern count in {city}?",
        "How many interns worked on Artificial Intelligence in {city}?"
    ], "Artificial Intelligence"),
    ([
        "How many tech interns worked on Full Stack Development in {city}?",
        "Full Stack Development intern count at {city}?",
        "Number of Full Stack interns in {city}?",
        "How many interns focused on Full Stack Development in {city}?"
    ], "Full Stack Development"),
    ([
        "How many tech interns worked on Data Science in {city}?",
        "Data Science intern count in {city}?",
        "Number of Data Science interns at {city}?",
        "How many interns worked on Data Science in {city}?"
    ], "Data Science"),
    ([
        "How many tech interns worked on Cybersecurity in {city}?",
        "Cybersecurity intern count at {city}?",
        "Number of Cybersecurity interns in {city}?",
        "How many interns focused on Cybersecurity in {city}?"
    ], "Cybersecurity"),
    ([
        "What is the name of the internship program offered by Infosys?",
        "Under what program do the interns work?",
        "What does Infosys call its internship program?",
        "Name of the internship initiative at Infosys?"
    ], "program_name"),
    ([
        "Which company offers the InStep Internship Program?",
        "What is the company behind the InStep program?",
        "Who runs the internship program?",
        "Which organization hosts this internship?"
    ], "company"),
    ([
        "What domains are covered by the InStep Internship Program?",
        "Which fields can interns specialize in?",
        "List the domains included in the internship.",
        "What areas do the interns work in?"
    ], "domains"),
    ([
        "What benefits are provided to InStep interns?",
        "Which perks do interns receive?",
        "What support is given to interns in the program?",
        "List the benefits available to interns."
    ], "benefits")
]

unanswerable_questions = [
    "Which universities did the interns graduate from?",
    "How many snacks do interns get each day?",
    "Did all interns publish research papers?",
    "Who was the CEO of Infosys in 2025?",
    "What was the WiFi password at the Hartford Hub?"
]

def generate_random_date(start_year=2019, end_year=2025):
    """Generate a random date between Jan 1 start_year and Dec 31 end_year"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta_days = (end_date - start_date).days
    random_days = random.randint(0, delta_days)
    return start_date + timedelta(days=random_days)

def generate_context(city, date, total_interns, ba_interns, tech_interns, tech_domains_counts, mentor):
    date_str = date.strftime("%B %d, %Y")
    context = (
        f"{base_info['company']} is a global leader in technology services and consulting. "
        f"The {base_info['program_name']} is an award-winning initiative offering real-world projects. "
        f"It spans domains including {', '.join(base_info['domains'])}. "
        f"Interns receive benefits such as {', '.join(base_info['benefits'])}. "
        f"On {date_str}, the {city} branch welcomed {total_interns} interns: "
        f"{ba_interns} in Business Analysis and {tech_interns} in Technology. "
        f"Of the Technology interns, {tech_domains_counts.get('Full Stack Development', 0)} worked on Full Stack Development, "
        f"{tech_domains_counts.get('Artificial Intelligence', 0)} on Artificial Intelligence, "
        f"{tech_domains_counts.get('Data Science', 0)} on Data Science, and "
        f"{tech_domains_counts.get('Cybersecurity', 0)} on Cybersecurity. "
        f"Interns were guided by mentors like {mentor}."
    )
    return context

def find_answer_start(context, answer):
    # Try multiple variants to find answer start index in context
    variants = [answer, answer.strip(), answer.lower(), answer.title()]
    for variant in variants:
        idx = context.find(variant)
        if idx != -1:
            return idx
    return -1

def generate_squad_format_json(num_samples, filename):
    data = []

    for _ in range(num_samples):
        city = fake.city()
        date = generate_random_date()
        total_interns = random.randint(10, 30)
        ba_interns = random.randint(2, max(2, total_interns // 3))
        tech_interns = total_interns - ba_interns

        domains = ["Full Stack Development", "Artificial Intelligence", "Data Science", "Cybersecurity"]
        tech_domains_counts = {}
        remaining = tech_interns
        for i, domain in enumerate(domains):
            if i == len(domains) - 1:
                tech_domains_counts[domain] = remaining
            else:
                val = random.randint(0, remaining)
                tech_domains_counts[domain] = val
                remaining -= val

        mentor = fake.name()
        context = generate_context(city, date, total_interns, ba_interns, tech_interns, tech_domains_counts, mentor)

        # Create answer dict for easy lookup
        answer_lookup = {
            "date": date.strftime("%B %d, %Y"),
            "total_interns": str(total_interns),
            "ba_interns": str(ba_interns),
            "tech_interns": str(tech_interns),
            "Artificial Intelligence": str(tech_domains_counts.get("Artificial Intelligence", 0)),
            "Full Stack Development": str(tech_domains_counts.get("Full Stack Development", 0)),
            "Data Science": str(tech_domains_counts.get("Data Science", 0)),
            "Cybersecurity": str(tech_domains_counts.get("Cybersecurity", 0)),
            "program_name": base_info["program_name"],
            "company": base_info["company"],
            "domains": ", ".join(base_info["domains"]),
            "benefits": ", ".join(base_info["benefits"]),
        }

        # Generate answerable questions
        for q_variants, key in question_templates:
            question_template = random.choice(q_variants)
            question = question_template.format(city=city)

            answer = answer_lookup.get(key)
            if answer is None:
                # Skip if no answer found (should not happen)
                continue

            answer_start = find_answer_start(context, answer)
            if answer_start == -1:
                # Sometimes the answer string might not be found exactly, skip those
                continue

            data.append({
                "context": context,
                "id": str(uuid.uuid4()),
                "question": question,
                "answers": {"text": [answer], "answer_start": [answer_start]},
                "is_impossible": False
            })

        # Generate unanswerable questions
        for uq in unanswerable_questions:
            data.append({
                "context": context,
                "id": str(uuid.uuid4()),
                "question": uq,
                "answers": {"text": [], "answer_start": []},
                "is_impossible": True
            })

    # Save as JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} QA examples to {filename}")


# Example usage:
generate_squad_format_json(1000, "infosys_instep_train.json")
generate_squad_format_json(250, "infosys_instep_val.json")
