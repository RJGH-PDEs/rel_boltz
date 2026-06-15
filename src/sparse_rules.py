import numpy as np

def andrea(select):
    # test function
    # k = select[0][0]
    l = select[0][1]
    # m = select[0][2]
    
    # f
    # k1 = select[1][0]
    l1 = select[1][1]
    # m1 = select[1][2]
    
    # g
    # k2 = select[2][0]
    l2 = select[2][1]
    # m2 = select[2][2]

    # rule 
    rule = l1 + l2 - l 

    return (0 <= rule) and (rule <= 2*min(l1, l2)) and (rule%2 == 0) 

def cai(select):
    # test function
    # k = select[0][0]
    # l = select[0][1]
    m = select[0][2]
    
    # f
    # k1 = select[1][0]
    # l1 = select[1][1]
    m1 = select[1][2]
    
    # g
    # k2 = select[2][0]
    # l2 = select[2][1]
    m2 = select[2][2]

    # rule 
    test = np.abs(m) 
    sum  = np.abs(m1 + m2)
    diff = np.abs(m1 - m2)

    return (test - sum == 0) or (test - diff == 0)

def andrea_test(select):
    print("select: ", select)
    print("andrea flag: ", andrea(select))
    return 0

# main funtion
def main():
    # choose the three functions
    # test function, phi
    k = 2
    l = 2
    m = 2
    
    # trial function f(p), no gradient
    k1 = 2
    l1 = 2
    m1 = -1
    
    # trial function \nabla g(p), gradient
    k2 = 2
    l2 = 2
    m2 = -1

    # package 
    select = [[k,l,m],[k1,l1,m1],[k2,l2,m2]]
    andrea_test(select)

# execute main funtion
if __name__ == "__main__":
    main()
