def fun(m):
    for i in range(m):
        yield i  

# call the generator function
print(list(fun(5)))
