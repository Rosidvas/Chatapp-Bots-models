

from Main import analyze_user_input

onchat = True
while(onchat):

    user_input = input("You:")
    response = analyze_user_input("Developer", user_input)
    print("Bot:", response)

