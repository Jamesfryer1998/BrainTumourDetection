import yagmail
import datetime

def email(type, run=None, conv=None, dense=None, combination=None):
    user = yagmail.SMTP(user='pythonemail1998@gmail.com', password='advdmdyqrynizeri')
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if type == 'run':
        user.send(to=('pythonemail1998@gmail.com')
                ,subject =f'Resolution {run} complete.',
                contents=f'Resolutions testing complete at {time}')
    elif type == 'conv_testing':
        user.send(to=(('pythonemail1998@gmail.com', 'chanatlive@outlook.com'))
                ,subject =f'Model Creation Conv:{conv} Dense:{dense}',
                contents=f'Model Structure test complete at {time}')

    elif type == 'hyper_testing':
         user.send(to=(('pythonemail1998@gmail.com', 'chanatlive@outlook.com'))
                ,subject =f'Hyperamater combination {combination}',
                contents=f'Hyperparamater test completed at {time}')



    