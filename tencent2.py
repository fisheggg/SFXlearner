#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param num string字符串 
# @return string字符串
#
class Solution:
    def maxLexicographical(self , num ):
        # write code here
        start = 0
        end = 0
        for i in range(0, len(num)-1):
            if num[i] == '0':
                start = i
                break

        for i in range(start+1, len(num)):
            if num[i] != '0':
                end = i
                break
            if i == len(num)-1 and num[i] == '0':
                end = i
                print('here')
                break

        out = num[0:start] + "1"*(end-start+1) + num[end+1:]
        return out