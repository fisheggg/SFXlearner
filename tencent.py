class Solution:
    def minM(self , n , k ):
        count = 0
        m = 0
        while count < n:
            m += 1
            st = self.convert(m, k)
            count += st.count('1')
        return m
    
    def convert(self, m, k):
        st = ""
        while m > 0:
            st += str(m % k)
            m = m // k
        return st[::-1]