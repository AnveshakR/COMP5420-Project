import praw
import pandas as pd


# Credentials Input Here!
user_agent = "Basic Scrapper"
reddit = praw.Reddit(
    client_id = "",
    client_secret = "",  
    user_agent = user_agent
)

# URL Input Here!
submission_url = ""

submission = reddit.submission(url=submission_url)
submission.comments.replace_more(limit=500)

reddit_text = []
reddit_time = []
reddit_upvote = []

# Scrapping Comments
count = 0
for comment in submission.comments.list():

    reddit_text.append(comment.body)
    reddit_time.append(comment.created_utc)
    reddit_upvote.append(comment.score)

    count = count + 1
    if count >= 2000:
        break

print("Scrapping Finish!")
# CSV Time!
csv = pd.DataFrame()
csv["text"] = reddit_text
csv["time"] = reddit_time
csv["upvote"] = reddit_upvote
output_csv_path = "output.csv"
csv.to_csv(output_csv_path, index=False)

print("CSV has been made!")