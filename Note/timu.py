# -*- coding: utf-8 -*-
# @Time    : 18-8-2 下午10:31
# @Author  : Marvin
# @File    : timu.py
# @Notes   : 


def main(text):
    text = str(text).split(' ')
    alpha = {}
    digit = {}
    for index, i in enumerate(text):
        if i.isalpha() is True:
            alpha[index] = i
        else:
            digit[index] = i
    for index, i in enumerate(alpha.keys()):
        text[i] = alpha[sort_by_value(alpha)[index]]
    for index, i in enumerate(digit.keys()):
        text[i] = digit[sort_by_value(digit)[index]]

    return ' '.join(text)


def sort_by_value(d):

    items = d.items()

    back_items = [[v[1], v[0]] for v in items]

    back_items.sort()

    return [back_items[i][1] for i in range(0, len(back_items))]


if __name__ == '__main__':
    print(main(1))
    print(main('car truck bus'))
    print(main('8 4 6 1 -2 9 5'.strip()))
    print(main('car truck 8 4 bus 6 1'))

    """
    You are to write a program that takes a list of strings containing integers and words and returns a sorted version of the list.

The goal is to sort this list in such a way that all words are in alphabetical order and all integers are in numerical order. Furthermore, if the nth element in the list is an integer it must remain an integer, and if it is a word it must remain a word.

Input Specification:
The input will contain a single, possibly empty, line containing a space-separated list of strings to be sorted. Words will not contain spaces, will contain only the lower-case letters a-z. Integers will be in the range -999999 to 999999, inclusive. The line will be at most 1000 characters long.

Output Specification:
The program must output the list of strings, sorted per the requirements above. Strings must be separated by a single space, with no leading space at the beginning of the line or trailing space at the end of the line.

Sample Input 1:
1
Sample Output 1:
1

Sample Input 2:
car truck bus
Sample Output 2:
bus car truck

Sample Input 3:
8 4 6 1 -2 9 5
Sample Output 3:
-2 1 4 5 6 8 9

Sample Input 4:
car truck 8 4 bus 6 1
Sample Output 4:
bus car 1 4 truck 6 8 
"""