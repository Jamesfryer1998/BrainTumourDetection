import yagmail
import datetime

def email(run=None, conv=None, dense=None):
    user = yagmail.SMTP(user='pythonemail1998@gmail.com', password='advdmdyqrynizeri')
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if conv == None and dense == None:
        user.send(to=('pythonemail1998@gmail.com')
                ,subject =f'Resolution {run} complete.',
                contents=f'Resolutions testing complete at {time}')
    else:
        user.send(to=(('pythonemail1998@gmail.com', 'chanatlive@outlook.com'))
                ,subject =f'Model Creation Conv:{conv} Dense:{dense}',
                contents=f'Model Structure test complete at {time}')