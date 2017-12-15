def get_host_path(HOST=False,PATH=True):
    import socket
    if socket.gethostname() =='x7' :
        host='x7'; path='/home/pachaya/HSMmodel/'
    elif socket.gethostname() =='g13':
        host='g13'; path= '/home/pachaya/AntolikData/SourceCode/'
    elif socket.gethostname() =='Pachayas-MacBook-Pro.local':
        host='localMBP'; path= '/Users/pachayasailamul/Documents/HSMmodel/'       
    else:
        #raise Exception('Unknown Host : Please add your directory at get_path()')
        import os
        host='Unknown'; path = os.getcwd() +'/'
    if(HOST&PATH):
        return host, path
    elif(HOST):
        return host
    else:
        return path
