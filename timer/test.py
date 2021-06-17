from timer import Timer

def func1(b):
    a=0
    for i in range(b):
        a +=1
    return b

if __name__=="__main__":
    for i in range(100):
        with Timer("first", timer=True):
            func1(10000)

        with Timer("second", timer=True):
            func1(20000)

        with Timer("third", timer=True):
            func1(30000)
        
        with Timer("fourth", timer=True):
            func1(40000)

    Timer().save()