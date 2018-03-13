# Correction for dqe. Does not agree with TEM-Sim at the moment

dqe = 0.4

mean = my_noisy_data.asarray()
shape = mean * (1 / (1-dqe))

data_after_dqe = np.random.wald(mean, shape)

data_after_dqe *= gain 

data_after_dqe = my_noisy_data.space.element(data_after_dqe)
data_after_dqe.show(title='my data after deq', coords = [0, None, None])
