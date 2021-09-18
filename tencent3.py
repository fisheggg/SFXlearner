# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param root TreeNode类 
# @param k int整型 
# @return TreeNode类
#
class Solution:
    def cyclicShiftTree(self , root , k ):
        # write code here
        if root is None:
            return None
        if root.left == None and root.right == None:
            return root

        for _ in(range(k)):
            a = None
            b = None
            c = None
            d = None
            if root.left:
                a = root.left.left
                b = root.left.right
            if root.right:
                c = root.right.left
                d = root.right.right
            e = root.left
            f = root.right
            if root.left:
                root.left.left = self.cyclicShiftTree(d, k)
                root.left.right = self.cyclicShiftTree(a, k)
            if root.right:
                root.right.left = self.cyclicShiftTree(b, k)
                root.right.right = self.cyclicShiftTree(c, k)
            root.left = f
            root.right = e
        return root