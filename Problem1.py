# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    m1 = -101
    m2 = -101
    for i in range(n):
        if arr[i] > m1:
            m2 = m1 
            m1 = arr[i]
        
        if (arr[i] > m2) & (arr[i] < m1) :
            m2 = arr[i]
    print(m2)
        

#Nested Lists

if __name__ == '__main__':
            
    m1 = 10000
    m2 = 10000
    N1 = []
    N2 = []
    for _ in range(int(raw_input())):
        name = raw_input()
        score = float(raw_input())
        if score == m1:
            N1.append(name)
        if score < m1:
            m2 = m1 
            m1 = score
            N2 = N1
            N1 = [name]  
        else:  
            if score == m2:
                N2.append(name)
            if (score < m2) & (score > m1) :
                m2 = score
                N2 = [name]

    alph_N2 = sorted(N2)
    for i in range (len(N2)):
        print(alph_N2[i])


#Validating Postal codes

regex_integer_in_range = r"\b[1-9][0-9]{5}\b"    # Do not delete 'r'.]
regex_alternating_repetitive_digit_pair = r"([0-9])(?=[0-9]\1)"    # Do not delete 'r'.

#Matrix script

#!/bin/python3

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
ma = []
for j in range(m):
    ma.append(''.join((matrix[i][j] for i in range(n))))
s = ''.join(ma)
#print(re.findall('([A-Z,a-z,0-9]){1}\W+([A-Z,a-z,0-9]){1}', s))
print(re.sub('([A-Z,a-z,0-9]){1}\W+([A-Z,a-z,0-9]){1}', r'\1 \2', s))
    

#Validating Credit Card Numbers

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())

for i in range(n):
    v = "Invalid"
    a = input()
    if (re.fullmatch(r'[4-6][0-9]{3}([-]?[0-9]{4}){3}', a) is not None): 
        if(re.search(r'(.)\1{3,}', a.replace('-', "")) is None):
            v = "Valid"
    print(v)

#Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())

for i in range(n):
    v = "Invalid"
    a = input()
    if re.fullmatch(r'[A-Z, a-z, 0-9]{10}', a) and  len(list(set(a))) == 10:
        if len(re.findall(r'[0-9]', a)) >= 3 and len(re.findall(r'[A-Z]', a)) >= 2:
            v = "Valid"
    print(v)


#validating-named-email-addresses

# Enter your code here. Read input from STDIN. Print output to STDOUT
import email.utils
import re
n = int(input())
symb = r'[a-zA-Z0-9][\w|-]{0,}[\w|.]{0,}@[a-zA-Z]+[.][a-zA-Z]{1,3}'
for i in range(n):
    a = input()
    e = email.utils.parseaddr(a)
    if re.fullmatch(symb, e[1]) is not None:
        
           print(a)


#python division

from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a//b)
    print(a/b)

#arithmetic operators

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)

#Finding the percentage
if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    scores = student_marks[query_name]
    print("%.2f"%(sum(scores)/len(scores)))


#Print Function

from __future__ import print_function

if __name__ == '__main__':
    n = int(raw_input())
    s = ''
    for i in range (n):
        s = s + str(i+1)
    print (s)





#py-hello-world

if __name__ == '__main__':
    print "Hello, World!"

#Write a function

def is_leap(year):
    leap = False
    if year%4 == 0:
        leap = True
        if year%100 == 0 and year%400 > 0:
            leap = False
    # Write your logic here
    
    return leap




#Loops
if __name__ == '__main__':
    n = int(raw_input())
    for i in range(n):
        print(i*i)


#What's your name

#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
# 2. STRING last
#

def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")
    return 


#swap Case
def swap_case(s):
    a=''
    for i in range(len(s)):
        if s[i] == s[i].upper():
           a =  a + s[i].lower()
        else:
           a = a + s[i].upper()
    return a


#Tuples
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
print(hash(tuple(integer_list)))






#Lists

if __name__ == '__main__':
    N = int(raw_input())
    a = []
    for i in range(N):
        inp = raw_input().split()
        inp[1:] = map(int, inp[1:])
        if inp[0] == 'print':
            print(a)
        if inp[0] == 'insert':
            if int(inp[1]) >= len(a):
                a.append(inp[2])
            else:
                a = a[:inp[1]] + [inp[2]] + a[(inp[1]):]
        if inp[0] == 'append':
            a.append(inp[1])
        if inp[0] == 'sort':
            a = sorted(a)
        if inp[0] == 'pop':
            a = a[:-1]
        if inp[0] == 'reverse':
            for t in range(len(a)//2):
                k = a[len(a) - t - 1]
                a[len(a) - t - 1] = a[t]
                a[t] = k
        if inp[0] == 'remove':
            ind = a.index(int(inp[1]))
            a = a[:(ind)] + a[(ind+1):]


#itertool.combinations():

# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations
if __name__ == '__main__':
    A, n = raw_input().split()
    for i in range(int(n)):
        Comb =  map(sorted, list(combinations(A,i+1)))
        C = sorted(Comb)
        for t in range(len(C)):
            print(''.join(C[t]))
        


#Capitalize!

# Complete the solve function below.
def solve(s):
    a = s.split(' ')
    for i in range(len(a)):
        if (len(a[i]) > 0):
            if a[i][0] == a[i][0].lower():
                a[i] =  a[i][0].upper() + a[i][1:]
    return ' '.join(a)




#Text Wrap



def wrap(string, max_width):
    wrapper = textwrap.TextWrapper(width=max_width,break_long_words=True)
    return '\n'.join(wrapper.wrap(string))


#Find a string

def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)-len(sub_string)+1):
        if string[i] == sub_string[0]:  
            if string[i:i+len(sub_string)] == sub_string:
                count+=1
    return count


#Mutations

def mutate_string(string, position, character):
    s_new = string[:position] + character + string[(position+1):]
    return s_new


#String Split and Join
def split_and_join(line):
    # write your code here
    line = "-".join(str(line).split(" "))
    return line



#the minion game
def minion_game(string):
    # your code goes here
    St = 0
    Kev = 0 
    winner = "Stuart"
    for i in range(len(string)):
        if string[i] in ["A","E","I","O","U"]:
            Kev += len(string) - i 
        else:
            St += len(string) - i 

    if Kev > St:
        winner = "Kevin"
    if Kev == St:
        print("Draw")
    else:
        print(f"{winner} {max(St, Kev)}")
    return 




#Zeros and Ones

import numpy as np
size = list(map(int, input().split()))
a = np.zeros(((size)), dtype = int)
b = np.ones(((size)), dtype = int)
print(a)
print(b)


#Concatenate

import numpy as np
N, M, P = input().split()
a = np.zeros((int(N), int(P)), dtype=int)
b = np.zeros((int(M), int(P)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
for i in range(int(M)):
     b[i, :] = input().split()
print (np.concatenate((a, b), axis = 0 ))

#Transpose and Flatten

import numpy as np
N, M = input().split()
a = np.zeros((int(N), int(M)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
print(np.transpose(a))
print(a.flatten())


#Arrays



def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array([(arr[len(arr) - k - 1]) for k in range(len(arr))],float)



#Sum and Prod
import numpy as np
N, M = input().split()
a = np.zeros((int(N), int(M)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
print(np.prod(np.sum(a, axis = 0)))




#Min and Max
import numpy as np
N, M = input().split()
a = np.zeros((int(N), int(M)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
print(np.max(np.min(a, axis=1)))



#floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')
a = np.array(list(map(float, input().split())))
print(np.floor(a))
print(np.ceil(a))
print(np.rint(a))



#Array Mathematics
import numpy as np
N, M = input().split()
a = np.zeros((int(N), int(M)), dtype=int)
b = np.zeros((int(N), int(M)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
for i in range(int(N)):
     b[i, :] = input().split()
print (np.add(a, b)) 

print (np.subtract(a, b)) 
print (np.multiply(a, b)) 
print (np.floor_divide(a, b)) 
print (np.mod(a, b)) 
print (np.power(a, b)) 


#Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')
a, b = input().split()
print (numpy.eye(int(a), int(b), k = 0))


#Linear Algebra

import numpy as np
N = input()
a = np.zeros((int(N), int(N)), dtype=float)
for i in range(int(N)):
     a[i, :] = input().split()
print (round(np.linalg.det(a), 2))




#Polynomials

import numpy
a = list(map(float, input().split()))
x = float(input())
print (numpy.polyval(a, x))


#Inner and Outer

import numpy as np
a = np.array(list(map(int, input().split())))
b = np.array(list(map(int, input().split())))
print(np.inner(a, b))
print(np.outer(a, b))


#Dot and cross

import numpy as np
N= input()
a = np.zeros((int(N), int(N)), dtype=int)
b = np.zeros((int(N), int(N)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
for i in range(int(N)):
     b[i, :] = input().split()
print (np.matmul(a, b))  
    


#Mean, Var and Std

import numpy as np
N, M = input().split()
a = np.zeros((int(N), int(M)), dtype=int)
for i in range(int(N)):
     a[i, :] = input().split()
print (np.mean(a, axis = 1))  
print (np.var(a, axis = 0))  
print (round(np.std(a, axis = None),11))   



#Set.union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = input()
a1 = set(list(map(int, input().split())))
s2 = input()
a2 = set(list(map(int, input().split())))
un = a1.union(a2)
print(len(un))



#Set.intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = input()
a1 = set(list(map(int, input().split())))
s2 = input()
a2 = set(list(map(int, input().split())))
inter = a1.intersection(a2)
print(len(inter))




#Set.add()

# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
st = set()
for i in range(n):
    country = input()
    st.add(country)
print(len(st))



#Symmetric Difference

# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = input()
a1 = set(list(map(int, input().split())))
s2 = input()
a2 = set(list(map(int, input().split())))
diff = a1.union(a2) - a1.intersection(a2)
for i in sorted(diff):
    print(i)



#Introduction to Sets  
def average(array):
    # your code goes here
    dist_arr = set(array)
    return sum(dist_arr)/len(dist_arr)

#check-strict-superset

a = set(list(map(int, input().split())))
n = int(input())
t = True
for i in range(n):
    b = set(list(map(int, input().split())))
    if a.intersection(b) != b or len(a) <= len(b):
        t = False
print (t)





#py-set-discard-remove-pop
n = int(input())
s = set(map(int, input().split()))
p = int(input())
for i in range(p):
    com = input().split()
    if com[0] == "pop":
        s.pop()
    if com[0] == "remove":
        s.remove(int(com[1])) 
    if com[0] == "discard":
        s.discard(int(com[1])) 
print(sum(s))



#No idea
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
s1, s2 = input().split()
h = 0
hap = list(map(int, input().split()))
a1 =set(list(map(int, input().split())))
a2 = set(list(map(int, input().split())))
for i in hap:
    if i in a1:
        h+=1
    else:
        if i in a2:
            h-=1
print(h)



#set-difference-operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = input()
a1 = set(list(map(int, input().split())))
s2 = input()
a2 = set(list(map(int, input().split())))
diff = a1.difference(a2)
print(len(diff))




#set-symmetric-difference-operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = input()
a1 = set(list(map(int, input().split())))
s2 = input()
a2 = set(list(map(int, input().split())))
diff = a1.union(a2) - a1.intersection(a2)
print(len(diff))


#merge-the-tools
def merge_the_tools(string, k):
    # your code goes here
    for i in range (int(len(string)/k)):
        el = string[i*k:(i+1)*k]
        j = 0
        while j < len(el)-1: 
            if el[len(el)-1 - j] in el[:len(el)-1 - j]:
                el = el[:len(el)-1 - j] + el[len(el)-1 - j+1:]
            else:
                j+=1
        print(el)
        


#string-validators
if __name__ == '__main__':
    s = input()
    alnum = False
    alpha = False
    digit = False
    upper = False
    lower = False
    for i in s:
        if i.isalnum():
            alnum = True
        if i.isalpha():
            alpha = True
        if i.isdigit():
            digit = True
        if i.isupper():
            upper = True
        if i.islower():
            lower =True
    print(alnum)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)
            

#py-the-captains-room
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
a = list(map(int, input().split()))
s = set(a)
for i in s:
    capt = 0
    for j in a:
        if i == j:
            capt += 1
        if capt > 1:
            break
    if capt == 1:
        print (i)
        break



#Set mutations

# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = input()
A = set(map(int, input().split()))
n = int(input())
for i in range(n):
    com = input().split()
    b = set(map(int, input().split()))
    if com[0] == "update":
        A.update(b)
    if com[0] == "intersection_update":
        A.intersection_update(b)
    if com[0] == "symmetric_difference_update":
        A.symmetric_difference_update(b) 
    if com[0] == "difference_update":
        A.difference_update(b) 
print(sum(A))


#check subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
for i in range(n):
    t = False
    s1 = input()
    a1 = set(list(map(int,input().split())))
    s2 = input()
    a2 = set(list(map(int,input().split())))
    if a1.intersection(a2) == a1:
        t = True
    print(t)



#Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
n = int(input())
d = OrderedDict()
for i in range(n):
    pr = input()
    if pr in d.keys():
        d[pr] += 1
    else: 
        d[pr] = 1
print(len(d))
print(*list(d.values()), ' ')




#defaultdict-tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
A = defaultdict(list)
n, m = input().split()
for i in range(int(n)):
    A[input()].append(str(i+1))
for i in range(int(m)):
    b = input()
    if b in A.keys():
        print(' '.join((A[b])))
    else:
        print(-1)
    

#py-collections-ordereddict
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
n = int(input())
d = OrderedDict()
for i in range(n):
    pr = input().split()
    if ' '.join(pr[:-1]) in d.keys():
        d[' '.join(pr[:-1])] += int(pr[len(pr) - 1])
    else: 
        d[' '.join(pr[:-1])] = int(pr[len(pr) - 1])
for key, value in d.items():
    print(key, value)


#py-collections-namedtuple
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
summ, n, st = 0, int(input()), namedtuple("st", input().split())
for i in range(n):
    summ += int(st._make(input().split()).MARKS)
print(round(summ/n, 2))





#collections-counter
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
l, size, n = int(input()), list(map(int, input().split())), int(input())
size_dict = Counter(size)
a = 0
for i in range(n):
    s_p = list(map(int, input().split()))
    if  size_dict[s_p[0]] > 0:
         size_dict[s_p[0]]-=1
         a+=s_p[1]
print(a)


#Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
m, d, y = list(map(int, input().split()))
print(list(calendar.weekheader(10).split())[calendar.weekday(y, m, d)].upper())


#Pilling up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
n = int(input())

for j in range(n):
    l = input()
    m = deque(map(int, input().split()))
    a = []
    for i in range (int(l)):
        if m[0] > m[-1]:
            a.append(m[0])
            m.popleft()
        else:
            a.append(m[-1])
            m.pop()
    if a == sorted(a, reverse = True):
        print("Yes")
    else:
        print("No")   


#collections-deque
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
d = deque()
n = int(input())
for i in range(n):
    c = input().split()
    if c[0] == "pop":
        d.pop()
    if c[0] == "append":
        d.append(c[1])
    if c[0] == "appendleft":
        d.appendleft(c[1])
    if c[0] == "popleft":
        d.popleft()
print(*d)


#Company Logo
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import OrderedDict


if __name__ == '__main__':
    s = input()
    n = len(s)
    d = OrderedDict()
    for i in range(n):
        pr = s[i]
        if pr in d.keys():
            d[pr] -= 1
        else: 
            d[pr] = -1    
    sort = sorted(d.items(), key = lambda x: (x[1],x[0]))
    for (key,value) in sort[:3]:
        print(key, abs(value))


#Athlete Sort
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []
    i = 0
    for j in range(n):
        arr.append(list(map(int, input().rstrip().split())))
        arr[j].append(i)
        i +=1
    k = int(input())
    for i in sorted(arr, key = lambda x: (x[k], x[m])):
        print(*(i[:-1]))


#Zipped
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = list(map(int, input().split()))
A=[]
for i in range(m):
    A = A+ [list(map(float, input().split()))]
Z = zip(*A)
for i in Z:
    print(sum(i)/m)


#map-and-lambda-expression
cube = lambda x: x*x*x# complete the lambda function 

def fibonacci(n):
    fib0 = [0, 1]
    fib = lambda i: fib0.append(sum(fib0[-2:]))
    if n>2:
        list(map(fib, [0]*(n-2)))
    else:
        fib0 = fib0[:n]
    return fib0
    # return a list of fibonacci numbers


#exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
for i in range(n):
    try:
        a, b = map(int, input().split())
        print (int(a//b))
    except ValueError as e:
        print ("Error Code:", e)
    except ZeroDivisionError as e:
        print ("Error Code:",e)


#time delta
#!/bin/python3

import math
import os
import random
import re
import sys
import time
from time import strptime
# Complete the time_delta function below.
import calendar
def time_delta(t1, t2):
    d1 = t1.split()
    d2 = t2.split()
    m1 =  int(strptime(d1[2],'%b').tm_mon)
    m2 = int(strptime(d2[2],'%b').tm_mon)
    m_d1 = 0
    m_d2 = 0
    y_d = (int(d1[3]) - int(d2[3]))*365 - int(calendar.leapdays(int(d1[3]), int(d2[3])))
    for i in range (m1-1):
        m_d1 += int(calendar.monthrange(int(d1[3]), i+1)[1])

    for i in range(m2-1):
        m_d2 += int(calendar.monthrange(int(d2[3]), i+1)[1])
    return str( abs(
(int (d1[1]) + m_d1 + y_d - int(d2[1]) - m_d2)*24*3600 +
(int(d1[4][:2]) - int(d2[4][:2]) - int(d1[5][:3]) + int(d2[5][:3]))*3600
 + (int(d1[4][3:5]) - int(d2[4][3:5]) - int(d1[5][3:5])*int(d1[5][0]+'1') + int(d2[5][3:5])*int(d2[5][0]+'1'))*60 + (int(d1[4][6:]) - int(d2[4][6:]))))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())
    for t_itr in range(t):
        t1 = input()

        t2 = input()
        d1 = t1.split()
        d2 = t2.split()
        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

#re.split()
regex_pattern = r"[, , . ]"	

#ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
s1 = sorted(input())
l = []
for i in s1:
    if i.islower():
        l.append(i)
for i in s1:
        if i.isupper():
            l.append(i)
for i in s1:
        if i.isdigit():
            if int(i)%2 == 1:
                l.append(i)
for i in s1:
        if i.isdigit():
            if int(i)%2 == 0:
                l.append(i)
print(''.join(l))


#validating-the-phone-number
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
symb = '[7-9]([0-9]){9}'
for i in range(n):
    if re.fullmatch(symb, input()) is None:
        print("NO")
    else:
        print("YES")

#Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
symb = '[+-]?([0-9]?)+\.[0-9]+'
for i in range(n):
    if re.fullmatch(symb, input()) is None:
        print("False")
    else:
        print("True")

