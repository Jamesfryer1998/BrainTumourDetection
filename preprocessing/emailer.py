import yagmail
import datetime

def email(run):
    #For this I made a new gmail and sent myself a email with the results in.
    user = yagmail.SMTP(user='pythonemail1998@gmail.com', password='advdmdyqrynizeri')

    #Features
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    user.send(to=('pythonemail1998@gmail.com')
            ,subject =f'Resolution {run} complete.',
            contents=f'Resolutions testing complete at {time}')