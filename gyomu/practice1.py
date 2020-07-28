# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:08:41 2020

@author: myy388
"""

print("hello world")

#name = input()
"""
変数埋め込み
print("my name is {}".format(name))
"""
#変数の応用
#print(f'age = {age}')#3.6以降


#ビット演算and, or　論理シフト
a = 12&21
b = 12|21
a = a<<1 #2倍
print(a,b)

#型表示
print(type(a))

#複素数j
i = 1 + 3j
k = 3 + 5j
print(i*k)
print(complex(3,5)*complex(1,3))

#文字列型
fruit = "apple"
print(fruit *10)
print(fruit[2])

#encode,decode
byte_fruit = fruit.encode('utf-8')
print(byte_fruit)
print(type(byte_fruit))

#startswith, strip(両端),rstrip(右のみ)
msg = 'ABCDEABC'
print(msg.startswith("ABCDE"))
msg = " ABC "
print(msg.rstrip())

#upper, lower, swapcase, replace, capitalize
msg = "abcABC"
msg_u = msg.upper()
msg_l = msg.lower()
msg_s = msg.swapcase()
msg_c = msg.capitalize()#1文字目のみ大文字
print(msg_u,msg_l,msg_s,msg_c)
print(msg.replace("ABC","FFF"))

#文字列一部取り出し
msg = "hello, my name is Taro"
print(msg[:5])
print(msg[::5])
print(msg[6:])
print(msg[1:10:2])
print(msg.islower())#全ての文字判定

#辞書型

