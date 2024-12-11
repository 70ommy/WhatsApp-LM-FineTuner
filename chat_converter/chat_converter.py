import csv
import re

def parse_chat_to_csv(input_file, output_file, question_name, answer_name):
    # regex for the chat format
    pattern = r'^(\d{2}/\d{2}/\d{2},\s*\d{2}:\d{2})\s*-\s*([^\:]+):\s*(.+)$'

    # read the file .txt
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # list for saving structured messages
    messages = []
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            timestamp, sender, content = match.groups()
            messages.append({"timestamp": timestamp, "sender": sender, "content": content})

    print(f"Total messages parsed: {len(messages)}")  # Debugging line

    # Create CSV file and set up writer
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['context', 'question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Initialize variables
        current_message = ""
        past_sender = None
        question = ""
        answer = ""
        context = []

        # Loop through the messages to classify questions and answers
        for i in range(len(messages)):
            current_message = messages[i]

            # Create context: Last 5 messages before the current one
            context = [msg['content'] for msg in messages[max(0, i-6):i-1]]

            # Check if the current sender is the question sender
            if current_message['sender'] == question_name:
                if past_sender == question_name:
                    # Append consecutive messages from the same person (question)
                    question += " " + current_message['content']
                else:
                    # Start a new question
                    question = current_message['content']
                past_sender = question_name

            elif current_message['sender'] == answer_name:
                if past_sender == answer_name:
                    # Append consecutive messages from the same person (answer)
                    answer += " " + current_message['content']
                else:
                    # Start a new answer
                    answer = current_message['content']
                past_sender = answer_name

            # Once we have a full question-answer pair, write to CSV
            if question and answer:
                writer.writerow({'context': " ".join(context), 'question': question, 'answer': answer})
                # Reset for next pair
                question = ""
                answer = ""

        # Final check in case the last message was a question-answer pair
        if question and answer:
            context = [msg['content'] for msg in messages[max(0, len(messages)-5):]]
            writer.writerow({'context': " ".join(context), 'question': question, 'answer': answer})

# Use the function
input_file = "assets/chat.txt"
output_file = "assets/chat.csv"
question_name = "Trainer"
answer_name = "Subject"
parse_chat_to_csv(input_file, output_file, question_name, answer_name)
