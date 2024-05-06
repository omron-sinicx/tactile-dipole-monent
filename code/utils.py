import inspect, time
import os.path
from pathlib import Path

FORMAT = {
   'PURPLE': '\033[95m',
   'CYAN': '\033[96m',
   'DARKCYAN': '\033[36m',
   'BLUE': '\033[94m',
   'GREEN': '\033[92m',
   'YELLOW': '\033[93m',
   'RED': '\033[91m',
   'BOLD': '\033[1m',
   'ITALIC': "\x1B[3m",
   'UNDERLINED': '\033[4m',
   'END': '\033[0m',
}

def get_caller_name(extension=False):
    caller_path = Path(inspect.stack()[2][1])
    caller_filename = caller_path.name
    return caller_filename if extension else os.path.splitext(os.path.basename(caller_filename))[0]

def log(*text, format=[], name_format=['italic','purple'], **kwargs):
    """ Modified version of print() that enables formatting and adds the speaker's name before the message """
    if isinstance(format, str): format = [format]
    format = ''.join([FORMAT[item.upper()] for item in format])
    text = ' '.join([f"{item}" for item in text])  
    if name_format not in ('ommit', 'OMMIT', None, False):
        filename = get_caller_name()
        if isinstance(name_format, str): name_format = [name_format]
        name_format = ''.join([FORMAT[item.upper()] for item in name_format])
        name = "|"+name_format+f"{filename}"+FORMAT['END']+"| "
    else: name=""
    text = format + (text.replace('\n', FORMAT['END']+'\n   '+' '*len(filename)+format) if name else text) + FORMAT['END']
    print(name+text, **kwargs)
    
def remaining_iter_ms(Ts, t0, infinite=1e20):
    t = max(1, int((Ts-(time.time()-t0))*1000))
    return t if t<infinite else 0  

def remaining_iter_s(Ts, t0):
    return max(0, Ts-(time.time()-t0))

class Info:
    """ Class to administer shared information between processes. Designed to be declared as a remote actor in ray. """
    def __init__(self, params={}):
        self._data = {}
        self._params = params
    
    def get(self, variable:str, key:str=""):
        """ Get atrtibute """
        var = getattr(self, '_'+variable)
        if key:
            if isinstance(var, dict) and key in var.keys():
                return var[key]
            else: return None
        else: return var.copy()
    
    def set(self, variable:str, newdata, overwrite=False):
        """ Set atrtibute """
        var = getattr(self, '_'+variable)
        if isinstance(var, dict) and not overwrite: var.update(newdata)
        else: var = newdata

class Exec:
    """ Envelope class that enables to pass an executable function while defining default arguments """
    def __init__(self, function, *default_args:list, **default_kwargs:dict) -> None:
        self.method = function
        self.default_args = list(default_args)
        self.default_kwargs = default_kwargs
    def __call__(self, *args, **kwargs):
        _kwargs = self.default_kwargs.copy()
        _kwargs.update(kwargs)
        _args = list(args)
        _args.extend(self.default_args[len(args):])
        return self.method(*_args, **_kwargs)

def progress_bar(n, N, title=''):  
    UP = '\033[1A'
    CLEAR = '\x1b[2K'
    p = (n+1)/N
    text = f"{int(p*100)}% "
    if title: title=title+' - '
    M = int(os.get_terminal_size().columns/1.5)-len(title+text)
    m = int(p*M)
    bar = "|"+"▆"*m + "_"*(M-m)+"|"
    log(title+text+bar, end='\r', flush=True)