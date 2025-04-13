import csv
import re

def parse_chat_to_csv(input_file, output_file, question_name, answer_name):
    # Regex to match WhatsApp message format: timestamp - sender: message
    pattern = r'^(\d{2}/\d{2}/\d{2},\s*\d{2}:\d{2})\s*-\s*([^:]+):\s*(.+)$'

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Parse messages
    messages = []
    for line in lines:

        match = re.match(pattern, line.strip())
        if match:
            timestamp, sender, content = match.groups()
            content = content.strip()
            if content == "<Media omessi>":
                content = "[media]"  # Replace media placeholder
            messages.append({"timestamp": timestamp, "sender": sender.strip(), "content": content})
            last_message_content = content
        else:
            if last_message_content:
                last_message_content += ". " + line.strip()
            else:   
                print(f"Line skipped (no match): {line.strip()}")

    print(f"Total messages parsed: {len(messages)}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['context', 'question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        question = ""
        answer = ""
        past_sender = None
        context_window = []

        for i, msg in enumerate(messages):
            sender = msg["sender"]
            content = msg["content"]

            # Build context from the previous 5 messages
            context_window = [m['content'] for m in messages[max(0, i - 5):i]]

            if sender == question_name:
                if past_sender == question_name or past_sender is None:
                    question += (" " if question else "") + content
                else:
                    question = content
                past_sender = question_name

            elif sender == answer_name:
                if past_sender == answer_name or past_sender is None:
                    answer += (" " if answer else "") + content
                else:
                    answer = content
                past_sender = answer_name

            # When a full question-answer pair is formed, write to CSV
            if question and answer:
                writer.writerow({
                    'context': " ".join(context_window),
                    'question': question.strip(),
                    'answer': answer.strip()
                })
                question = ""
                answer = ""

        # Final check for any remaining Q/A pair
        if question and answer:
            writer.writerow({
                'context': " ".join(context_window),
                'question': question.strip(),
                'answer': answer.strip()
            })

# Use the function
input_file = "assets/chat.txt"
output_file = "assets/chat.csv"
question_name = "Trainer"
answer_name = "Subject"
parse_chat_to_csv(input_file, output_file, question_name, answer_name)
