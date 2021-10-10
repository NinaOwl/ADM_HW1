# birthday-cake-candles https://www.hackerrank.com/challenges/birthday-cake-candles/problem

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    maxim = max(candles)
    count = 0
    for i in range(len(candles)):
        if candles[i] == maxim:
            count +=1
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()



#kangaroo https://www.hackerrank.com/challenges/kangaroo/problem

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here
    result = "NO"
    if (x1-x2)*(v2-v1) > 0:
        if (x1-x2)%(v2-v1) ==0:
            result = "YES"
    if x1 == x2 and v1 == v2:
        result = "YES"
    return (result)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()





#viral advertising https://www.hackerrank.com/challenges/strange-advertising/problem

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    r = 5
    likes = 0
    for i in range(n):
        likes = likes + r//2
        r = r//2 * 3
    # Write your code here
    return likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()






#Digit sum https://www.hackerrank.com/challenges/recursive-digit-sum/problem

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    # Write your code here
    l = int(n)
    summ = l%10
    l = l//10
    while l > 0:
        summ += l%10
        l = l//10
    l = summ * k 
    while l > 9:
        summ = 0
        while l > 0:
            summ += l%10
            l = l//10
        l = summ
    return l
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()






#InsertionSort1 https://www.hackerrank.com/challenges/insertionsort1/problem

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    el = arr[n - 1]
    i = n - 1
    while arr[i - 1] >= el:
        print(*(arr[:(i-1)+1] + [arr[i-1]] + arr[(i-1)+1:(n-1)]), sep = " ")
        i-=1
        if i < 1:
            break
    print (*(arr[:(i-1)+1] + [el] + arr[(i-1)+1:(n-1)]), sep = " ")
    # Write your code here

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)



#InsertionSort2 https://www.hackerrank.com/challenges/insertionsort2/problem

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    #print(n-1)
    #print(arr)
    el = arr[n - 1]
    i = n - 1
    while arr[i - 1] >= el:
        #arr = (arr[:(i-1)+1] + [arr[i-1]] + arr[(i-1)+1:(n-1)]), sep = " ")
        i-=1
        if i < 1:
            break
    return (arr[:i] + [el] + arr[i:n-1] + arr[n:len(arr)])
    
def insertionSort2(n, arr):
    for i in range(n-1):
        arr = insertionSort1(i+2, arr)
        print(*arr)
    # Write your code here
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)


