import yagmail
import datetime

def email(type, run=None, conv=None, dense=None, combination=None, rmse=None, params=None, run_time=None):
    user = yagmail.SMTP(user='pythonemail1998@gmail.com', password='YOUR_PASSWORD_HERE')
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if type == 'run':
        user.send(to=('pythonemail1998@gmail.com')
                ,subject =f'Resolution {run} complete.',
                contents=f'Resolutions testing complete at {time}')
    elif type == 'conv_testing':
        user.send(to=(('pythonemail1998@gmail.com'))
                ,subject =f'Model Creation Conv:{conv} Dense:{dense}',
                contents=f'Model Structure test complete at {time}')

    elif type == 'hyper_testing':
         user.send(to=(('pythonemail1998@gmail.com'))
                ,subject =f'Hyperamater combination {combination}',
                contents=f'Hyperparamater test completed at {time}')

    elif type == 'random_search':
        user.send(to=(('pythonemail1998@gmail.com'))
                ,subject =f'Random Search Completed on run: {run}',
                contents=f'Rmse: {rmse}   Params" {params}    Total run time: {run_time}')