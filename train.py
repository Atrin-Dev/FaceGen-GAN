from generator import generator
from critic import critic
from utils import *

# Dataset
data_path='./data/celeba/img_align_celeba'
ds = Dataset(data_path, size=128, lim=10000)

# DataLoader
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=12)


# Models
gen = generator(z_dim).to(device)
crit = critic().to(device)

# Optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5,0.9))
crit_opt= torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5,0.9))

"""
# Initializations
# gen=gen.apply(init_weights)
# crit=crit.apply(init_weights)
"""

# Training loop
for epoch in range(n_epochs):
    for real, _ in tqdm(dataloader):
        cur_bs = len(real)  # 128
        real = real.to(device)

        # CRITIC
        mean_crit_loss = 0
        for _ in range(crit_cycles):
            crit_opt.zero_grad()

            noise = gen_noise(cur_bs, z_dim)
            fake = gen(noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            alpha = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)  # 128 x 1 x 1 x 1
            gp = get_gp(real, fake.detach(), crit, alpha)

            crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp

            mean_crit_loss += crit_loss.item() / crit_cycles

            crit_loss.backward(retain_graph=True)
            crit_opt.step()

        crit_losses += [mean_crit_loss]

        # GENERATOR
        gen_opt.zero_grad()
        noise = gen_noise(cur_bs, z_dim)
        fake = gen(noise)
        crit_fake_pred = crit(fake)

        gen_loss = -crit_fake_pred.mean()
        gen_loss.backward()
        gen_opt.step()

        gen_losses += [gen_loss.item()]


        if cur_step % save_step == 0 and cur_step > 0:
            print("Saving checkpoint: ", cur_step, save_step)
            save_checkpoint(epoch, cur_step/save_step)

        if (cur_step % show_step == 0 and cur_step > 0):
            show(fake, wandbactive=1, name='fake')
            show(real, wandbactive=1, name='real')

            gen_mean = sum(gen_losses[-show_step:]) / show_step
            crit_mean = sum(crit_losses[-show_step:]) / show_step
            print(f"Epoch: {epoch}: Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")

            plt.plot(
                range(len(gen_losses)),
                torch.Tensor(gen_losses),
                label="Generator Loss"
            )

            plt.plot(
                range(len(gen_losses)),
                torch.Tensor(crit_losses),
                label="Critic Loss"
            )

            plt.ylim(-150, 150)
            plt.legend()
            plt.show()

        cur_step += 1