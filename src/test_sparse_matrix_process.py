import sparse_matrix_process as smp
import nose.tools as n

def test_get_hashmap():
    vec = [2.3,4.0,2.5]
    res = smp.get_hashmap(vec)
    n.assert_equal(res, {0:2.3,1:4.0,2:2.8})

def test_symmetrize():
    mat = [[1,2,3],[1,2,0],[1,0,0]]
    res = smp.symmetrize(mat)
    n.assert_equal(res, [[1,2,3],[1,2,3][1,1,1]])

def test_get_user_hashmap():
    user = [[2.3,4.0,2.5],[2.4,4.0,2.6],[2.2,4.0,2.8]]
    res = smp.get_user_hashmap(user)
    n.assert_equal(res, [{0:2.3,1:4.0,2:2.8},{0:2.4,1:4.0,2:2.6},{0:2.4,1:4.0,2:2.8}] )

def test_pearson_sparse_sim():
    user1 = [1,2,3,4]
    user2 = [1,2,3,4]
    user3 = [4,3,2,1]
    res1 = smp.pearson_sparse_sim(user1,user2)
    res2 = smp.pearson_sparse_sim(user1,user3)
    n.assert_equal(res1, 1.0 )
    n.assert_equal(res2, 0.0 )

def test_adj_adj_pearson_sim():
    user1 = [1,2,3,4]
    user2 = [1,2,3,4]
    user3 = [4,3,2,1]
    common = 5
    res1 = smp.adj_pearson_sim(user1,user2,common)
    res2 = smp.adj_pearson_sim(user1,user3,common)
    n.assert_equal(res1, 0.8 )
    n.assert_equal(res2, 0.0 )
