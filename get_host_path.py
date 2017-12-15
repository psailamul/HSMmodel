def get_host_path(HOST=False,PATH=True):
    import socket
    if socket.gethostname() =='x7' :
        host='x7'; path='/home/pachaya/HSMmodel/'
    elif socket.gethostname() =='g13':
        host='g13'; path= '/home/pachaya/AntolikData/SourceCode/'
    elif socket.gethostname() =='x8':
        host='x8'; path= '/home/pachaya/HSMmodel/'
    else:
        raise Exception('Unknown Host : Please add your directory at get_path()')
    if(HOST&PATH):
        return host, path
    elif(HOST):
        return host
    else:
        return path
