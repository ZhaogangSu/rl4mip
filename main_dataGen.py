from ml4co.DataGenerator.MIPDataGen import MIPDataGen

# 两种方法：'L2O', 'ACM'
dataGenerator = MIPDataGen(method='L2O', num_workers=10)

# 第一步：处理数据
dataGenerator.preprocess(dataset='setcover', num_train=5)

# 第二部：训练模型
dataGenerator.train(dataset='setcover', cuda=0, steps=8, save_start=3, save_step=2, num_samples=3)

# 第三步：生成新的数据
dataGenerator.generate(num_samples=8, mask_ratio=0.01)
