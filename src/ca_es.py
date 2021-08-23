import copy
import logging
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import random
import time
import torch
import torch.multiprocessing as tmp
import torch.nn.functional as F
import torch.tensor as tt
from torchvision.utils import save_image
from dist import Master, Worker
from net import CAModel
from pool import CustomPool
from utils import load_emoji, to_rgb, visualize_batch, append_file, write_file, export_model, dmg
from weight_updates import hebbian_update

HIDDEN_SIZE = None


class EvolutionStrategy:
    """Master class for performing an evolution. 
        Keeps track of hyperparameters, weights/coeffs.
        Contains methods for running the environment, evaluate performances and update parameters.
    """
    def __init__(self, args):
        self.iterations = args.iter
        self.learning_rate = args.lr
        self.sigma = args.sigma
        self.pop_size = args.pop_size
        self.fire_rate = args.fire_rate
        self.target_size = args.size
        self.target_padding = args.pad
        self.new_size = self.target_size + 2 * self.target_padding
        self.channel_n = args.channels
        self.hidden_size = args.hidden_size
        HIDDEN_SIZE = self.hidden_size
        self.target_img = load_emoji(args.emoji, self.target_size)
        self.use_hebb = args.hebb
        self.use_pool = args.pool
        self.damage = args.damage
        self.damageChannels = args.damageChannels
        self.use_mp = args.use_mp
        self.decay_state = 0
        self.log_main_every = 10
        self.hit_goal = False

        self.cross_machine = args.cross_machine
        self.is_master = args.master
        self.nodes = args.nodes
        
        if self.damage > 0:
            if not self.use_pool and not self.damage <=3:
                raise ValueError("use_pool needs to be true and damage_bottom_n < 4.")

        if self.cross_machine:
            if self.is_master:
                self.master = Master(nodes=args.nodes)
            else:
                self.worker = Worker(run_id=0)

        p = self.target_padding
        self.pad_target = F.pad(tt(self.target_img), (0, 0, p, p, p, p))
        h, w = self.pad_target.shape[:2]
        self.seed = np.zeros([h, w, self.channel_n], np.float64)
        self.seed[h // 2, w // 2, 3:] = 1.0
        
        if self.use_pool:
            self.pool_size = 1024
            self.batch_size = 4
            self.pool = CustomPool(self.seed, self.pool_size)

        else:
            self.batch_size = 1

        if self.use_hebb:
            self.coefficients_per_synapse = 5
            plastic_weights = 3 * self.channel_n * self.hidden_size + self.hidden_size * self.channel_n
            self.coeffs_start_interval = 0.001
            self.coeffs = np.random.uniform(-self.coeffs_start_interval, self.coeffs_start_interval,
                                            (plastic_weights, self.coefficients_per_synapse))
            self.net = CAModel(channel_n=self.channel_n, fire_rate=self.fire_rate, new_size_pad=self.new_size,
                               disable_grad=True, hidden_size=self.hidden_size, batch_size=self.batch_size, use_hebb=True)

        else:
            self.net = CAModel(channel_n=self.channel_n, fire_rate=self.fire_rate, new_size_pad=self.new_size,
                               disable_grad=True, hidden_size=self.hidden_size, batch_size=self.batch_size)
            self.parameters_shape = [tuple(w.shape) for w in self.net.parameters()]
        
        self.log_folder = args.log_folder
        logging.basicConfig(filename=self.log_folder + "/logging.txt", format='%(message)s', filemode="w",
                            level=logging.INFO)

        if args.pre_trained != "":
            if self.use_hebb:
                self.coeffs = np.load(args.pre_trained)
            else:
                self.load_model(args.pre_trained)

        logging.info("lr/(pop*sigma) at start: " + str(self.learning_rate / (self.pop_size * self.sigma)))
        
        # For logging
        self.x_range = []
        self.y_lin = []
        self.avg = []
        self.avg_iter = []
        self.losses_main = []
        self.iter_main = []

        t_rgb = to_rgb(self.pad_target).permute(2, 0, 1)
        save_image(t_rgb, self.log_folder + "/target_image.png")

    def load_model(self, path):
        """Load a PyTorch model from path."""
        self.net.load_state_dict(torch.load(path))
        self.net.double()

    def fitness_shaping(self, x):
        """Sort x and and map x to linear values between -0.5 and 0.5
            Return standard score of x
        """
        shaped = np.zeros(len(x))
        shaped[x.argsort()] = np.arange(len(x), dtype=np.float64)
        shaped /= (len(x) - 1)
        shaped -= 0.5
        shaped = (shaped - shaped.mean()) / shaped.std()
        return shaped

    def update_coeffs(self, fitnesses, epsilons):
        """Update parent Hebbian coefficients using evaluated mutants and fitness."""
        fitnesses = self.fitness_shaping(fitnesses)

        for index, c in enumerate(self.coeffs):
            layer_population = np.array([p[index] for p in epsilons])

            update_factor = self.learning_rate / (self.pop_size * self.sigma)
            self.coeffs[index] = c + update_factor * np.dot(layer_population.T, fitnesses).T

    def update_parameters(self, fitnesses, epsilons):
        """Update parent network weights using evaluated mutants and fitness."""
        fitnesses = self.fitness_shaping(fitnesses)

        for i, e in enumerate(epsilons):
            for j, w in enumerate(self.net.parameters()):
                w.data += self.learning_rate * 1 / (self.pop_size * self.sigma) * fitnesses[i] * e[j]

    def get_population(self, use_seed=None):
        """Return an array with values sampled from N(0, sigma).
            The shape of the array is (pop_size, (layer1_size, layer2_size)) using ES and  (pop_size, plastic_weights, 5)
        """
        if use_seed is not None:
            np.random.seed(use_seed)

        temp_pop = self.pop_size
        if self.is_master:
            temp_pop /= self.nodes
        eps = []
        if self.use_hebb:
            layers = self.coeffs
            for i in range(int(temp_pop / 2)):
                e = []
                e2 = []
                for w in layers:
                    j = np.random.randn(*w.shape) * self.sigma
                    e.append(j)
                    e2.append(-j)
                eps.append(e)
                eps.append(e2)
        else:
            layers = self.parameters_shape
            for i in range(int(temp_pop / 2)):
                e = []
                e2 = []
                for w in layers:
                    j = np.random.randn(*w) * self.sigma
                    e.append(j)
                    e2.append(-j)
                eps.append(e)
                eps.append(e2)
        return np.array(eps, dtype=np.object)

    def train_step_hebb(self, model_try, coeffs_try, x):
        """Perform a generation of CA. Initialize a random net and update weights in every update step using
            trained coeffs.
            Return output x and loss 
        """
        torch.seed()
        losses = torch.zeros(x.shape[0])
        for j, x0 in enumerate(x): # Iterate over batch
            model_try.apply(weights_init)
            model_try.fc1.weight.zero_()
            x0 = x0[None, ...]
            weights1_2, weights2_3 = list(model_try.parameters())
            weights1_2 = weights1_2.detach().numpy()
            weights2_3 = weights2_3.detach().numpy()

            iter_n = torch.randint(30, 40, (1,)).item() # Episode
            for i in range(iter_n):
                o0, o1, x0 = model_try(x0)
    
                weights1_2, weights2_3 = hebbian_update(coeffs_try, weights1_2, weights2_3, o0.numpy(),
                                                        o1.numpy(), x0.numpy())

                (a, b) = (0, 1)
                list(model_try.parameters())[a].data /= list(model_try.parameters())[a].__abs__().max()
                list(model_try.parameters())[b].data /= list(model_try.parameters())[b].__abs__().max()
                list(model_try.parameters())[a].data *= 0.4
                list(model_try.parameters())[b].data *= 0.4 

            loss = model_try.loss_f(x0, self.pad_target)
            loss = torch.mean(loss)
            losses[j] = loss.item()
            x[j] = x0[0, ...]

        loss = torch.mean(losses)
        return x, loss.item()

    def train_step_es(self, model_try, x):
        """Perform a generation of CA using trained net.
            Return output x and loss 
        """
        torch.seed()

        iter_n = torch.randint(30, 40, (1,)).item()

        for i in range(iter_n): # Episode
            x = model_try(x)

        loss = self.net.loss_f(x, self.pad_target)
        loss = torch.mean(loss)

        return x, loss.item()

    def get_fitness_hebb(self, epsilon, x0, pid, q=None):
        """Method that start a generation of Hebbian ES.
            Return output from generation x and its fitness
        """
        model_try = CAModel(channel_n=self.channel_n, fire_rate=self.fire_rate, new_size_pad=self.new_size,
                               disable_grad=True, hidden_size=self.hidden_size, batch_size=self.batch_size, use_hebb=True)
        torch.seed()
        model_try.apply(weights_init)
        coeffs_try = self.coeffs.copy()
        coeffs_try += epsilon
        
        x, loss = self.train_step_hebb(model_try, coeffs_try, x0.clone())
        fitness = -loss

        if not math.isfinite(fitness):
            raise ValueError('Fitness ' + str(fitness) + '. Loss: ' + str(loss))
        if self.use_mp:
            q.put((x, fitness, pid))
            return
        return x, fitness

    def get_fitness_es(self, epsilon, x0, pid, q=None):
        """Method that start a generation of ES.
            Return output from generation x and its fitness
        """
        model_try = copy.deepcopy(self.net)
        if epsilon is not None:
            for i, w in enumerate(model_try.parameters()):
                w.data += torch.tensor(epsilon[i])

        x, loss = self.train_step_es(model_try, x0)
        fitness = -loss
        
        if not math.isfinite(fitness):
            raise ValueError('Encountered non-number value in loss. Fitness ' + str(fitness) + '. Loss: ' + str(loss))
        if self.use_mp:
            q.put((x, fitness, pid))
            return
        return x, fitness

    def evaluate_main(self, x0):
        """Return output and fitness from a generation using unperturbed weights/coeffs"""
        if self.use_hebb:
                x_main, loss_main = self.train_step_hebb(self.net, self.coeffs, x0.clone())
                fit_main = - loss_main
        else:
                x_main, loss_main = self.train_step_es(self.net, x0.clone())
                fit_main = - loss_main
        return x_main, fit_main

    def create_plots(self, x_range, y_lin, avg_iter, avg, iter_main, losses_main):
        """Plot population's fitnesses, average fitnesses and main network's fitnesses.
            Two plots, one for all iterations so far, and one for the last 100 iterations.
        """
        plt.clf()
        plt.scatter(x_range, np.log10(y_lin), color="blue", s=0.5)
        plt.plot(avg_iter, np.log10(avg), color='pink')
        plt.plot(iter_main, np.log10(losses_main), color='red', alpha=0.7)
        plt.title("Log-loss for " + self.log_folder)
        plt.savefig(self.log_folder + "/log_loss_over_time.png")

        if len(x_range) >= 100 * self.pop_size:
            # log 10, last 100 iters
            plt.clf()
            plt.scatter(x_range[-100 * self.pop_size:], np.log10(y_lin[-100 * self.pop_size:]), s=0.5)
            plt.plot(avg_iter[-100:], np.log10(avg[-100:]), color='red')
            plt.title("Log-loss last 100 for " + self.log_folder)
            plt.savefig(self.log_folder + "/log_loss_over_time_last100.png")

    def save_data(self, buffer, x_range, y_lin, iter_main, losses_main, iteration):
        """Save raw population and main network fitnesses to a csv file on the format: iteration, fitness"""
        if len(x_range) > 0:
            points = buffer * self.pop_size
            append_file(self.log_folder + '/raw/losses.csv', x_range[-points:], y_lin[-points:])
        # this one overwrites
        write_file(self.log_folder + '/raw/main_losses.csv', iter_main, losses_main)
        if self.use_hebb:
            np.save(self.log_folder + "/models/" + str(iteration) + '.npy', self.coeffs)
        else:
            export_model(self.net, self.log_folder + "/models/saved_model_" + str(iteration) + ".pt")

    def log(self, fitnesses, iteration, x0=None, xs=None):
        """Function to add fitnesses to arrays and plot/save data at iteration intervals."""
        if x0 is None:
            x0 = tt(np.repeat(self.seed[None, ...], self.batch_size, 0))

        # Logging/plotting
        for k, fit in enumerate(fitnesses):
            self.x_range.append(iteration)
            self.y_lin.append(-fit)
        self.avg.append(-np.average(fitnesses))
        self.avg_iter.append(iteration)
        
        # Evaluate main net/coeffs 
        if iteration % self.log_main_every == 0:
            x_main, fit_main = self.evaluate_main(x0.clone())
            self.losses_main.append(-fit_main)
            self.iter_main.append(iteration)

        # Visualize batch and plot points
        if iteration % 500 == 0:
            if xs == None:
                visualize_batch([x_main], iteration, self.log_folder, nrow=self.batch_size)
            else:
                selected = xs[np.argmax(fitnesses)]
                visualize_batch([x0.clone(), selected, x_main], iteration, self.log_folder, nrow=self.batch_size)
            self.create_plots(self.x_range, self.y_lin, self.avg_iter, self.avg, self.iter_main, self.losses_main)
            
        # Save points and weights/coeffs to file
        buffer = 1000
        if iteration % buffer == 0:
            self.save_data(buffer, self.x_range, self.y_lin, self.iter_main, self.losses_main, iteration)
        
        mean_fit = np.mean(fitnesses)

        # Decay learning rate
        if mean_fit >= -0.03 and self.decay_state == 0:
            self.learning_rate *= 0.3
            self.decay_state += 1
            logging.info("Setting lr to " + str(self.learning_rate) + " at iter " + str(iteration))
        elif mean_fit >= -0.01 and self.decay_state == 1:
            self.learning_rate *= 0.5
            self.decay_state += 1
            logging.info("Setting lr to " + str(self.learning_rate) + " at iter " + str(iteration))

        print('step: %d, mean fitness: %.3f, best fitness: %.3f' % (iteration, mean_fit, np.max(fitnesses)))
        
        # check = 250
        # if (len(self.losses_main) > check//self.log_main_every) and not self.hit_goal:
        #     mean_main_loss = np.mean(self.losses_main[-(check//self.log_main_every):])
        #     if mean_main_loss <= 0.001:
        #         logging.info("Hit goal at " + str(iteration))
        #         if self.use_hebb:
        #             np.save(self.log_folder + "/models/" + str(iteration) + "good" + '.npy', self.coeffs)
        #         else:
        #             export_model(self.net, self.log_folder + "/models/saved_model_" + str(iteration) + "good" + ".pt")
        #         self.hit_goal = True

    def run_master(self):
        """Send weights/coeffs to worker nodes and poll for results.
            Update weights/coeffs when all results are present.
        """
        # ticM = time.time()
        for iter in range(self.iterations):
            # logging.info("Sending weights")
            weights_to_send = self.coeffs if self.use_hebb else self.net.state_dict()
            self.master.send_weights(weights_to_send)

            # logging.info("Waiting for results...")
            fitnesses, seeds = self.master.wait_for_results()
            # logging.info("Got all results!")
    
            fitnesses = np.array(fitnesses)
            eps_seeds = np.array(seeds)
            epsilons = []
            for seed in eps_seeds:
                eps = self.get_population(use_seed=seed)
                epsilons.append(eps)

            for i, fit in enumerate(fitnesses):
                if self.use_hebb:
                    self.update_coeffs(fit, epsilons[i])
                else:
                    self.update_parameters(fit, epsilons[i])
              
            all_fitnesses = []
            for fit in fitnesses:
                all_fitnesses.extend(fit)
            self.log(all_fitnesses, iter)
            # if iter == 999:
            #     tocM = time.time()
            #     logging.info("time used in milliseconds: " + str(int((tocM -ticM)*1000)))

    def run(self):
        """Start evolution using Hebbian or ES.
            If using multiple nodes this method will listen for weights/coeffs and send results.
            If not the method will also start other methods to update parameters and log results.

            If using a pool the method will sample x's from the pool and damage them (if damage is enabled), before every generation.
        """
        # seed
        x0 = tt(np.repeat(self.seed[None, ...], self.batch_size, 0))

        # Run models once to compile jitted methods.
        if self.use_hebb:
            model_try = CAModel(channel_n=self.channel_n, fire_rate=self.fire_rate, new_size_pad=self.new_size,
                            disable_grad=True, hidden_size=self.hidden_size, batch_size=self.batch_size, use_hebb=True)
            _, _ = self.train_step_hebb(model_try, self.coeffs, x0.clone())
        else:
            _, _ = self.train_step_es(self.net, x0.clone())

        if self.use_mp:
            processes = []
            q = tmp.Manager().Queue()

        for iter in range(self.iterations):
            if self.use_pool:
                batch = self.pool.sample(self.batch_size)
                x0 = batch["x"]
                loss_rank = self.net.loss_f(tt(x0), self.pad_target).numpy().argsort()[::-1]
                x0 = x0[loss_rank]
                x0[:1] = self.seed
                if self.damage:
                    for i in range(self.damage):
                        x0[-(i+1)] = dmg(x0[-(i+1)], self.new_size, only_bottom=True)

                x0 = tt(x0)

            if self.cross_machine:
                if self.use_hebb:
                    self.coeffs = self.worker.poll_weights()
                else:
                    weights = self.worker.poll_weights()
                    self.net.load_state_dict(weights)

                eps_seed = np.random.randint(0, 2**32-1)
                epsilons = self.get_population(use_seed=eps_seed)
            else:
                epsilons = self.get_population()
            
            fitnesses = np.zeros((self.pop_size), dtype=np.float64)
            xs = torch.zeros(self.pop_size, *x0.shape, dtype=torch.float64)
            for i in range(self.pop_size):
                if self.use_hebb:
                    if self.use_mp:
                        p = tmp.Process(target=self.get_fitness_hebb, args=(np.array(epsilons[i], dtype=np.float64), x0.clone(), i, q))
                        processes.append(p)
                    else:
                        x, fit = self.get_fitness_hebb(np.array(epsilons[i], dtype=np.float64), x0.clone(), i)
                else:
                    if self.use_mp:
                        p = tmp.Process(target=self.get_fitness_es, args=(epsilons[i], x0.clone(), i, q))
                        processes.append(p)
                    else:
                        x, fit = self.get_fitness_es(epsilons[i], x0.clone(), i)
                        
                if not self.use_mp:
                    fitnesses[i] = fit
                    xs[i] = x
                
            if self.use_mp:
                for p in processes:
                    p.start()
                
                for p in processes:
                    p.join()
                    x, fit, pid = q.get()
                    fitnesses[pid] = fit
                    xs[pid] = x

                processes = []
                if not q.empty():
                    print("Queue not empty")
            
            if self.use_pool:
                idx = np.argmax(fitnesses)
                batch["x"][:] = xs[idx]
                self.pool.commit(batch)
            
            fitnesses = np.array(fitnesses).astype(np.float64)
            if self.cross_machine:
                self.worker.send_result(fitnesses, eps_seed)
            else:
                if self.use_hebb:
                    self.update_coeffs(fitnesses, epsilons)
                else:
                    self.update_parameters(fitnesses, epsilons)

                self.log(fitnesses, iter, x0=x0, xs=xs)

            
def weights_init(m):
    """Initialize a network's weights with uniform distributed values.
        Used for Hebbian ES.
    """
    if isinstance(m, torch.nn.Linear) and m.in_features != HIDDEN_SIZE:
        torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)