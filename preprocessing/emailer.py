import yagmail
import datetime

def email(run):
    user = yagmail.SMTP(user='pythonemail1998@gmail.com', password='advdmdyqrynizeri')
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    user.send(to=('pythonemail1998@gmail.com')
            ,subject =f'Resolution {run} complete.',
            contents=f'Resolutions testing complete at {time}')