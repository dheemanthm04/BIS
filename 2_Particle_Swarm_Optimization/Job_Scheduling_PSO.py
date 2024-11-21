import numpy as np

# Problem Parameters
NUM_JOBS = 6
NUM_MACHINES = 3
PROCESSING_TIMES = np.random.randint(1, 10, (NUM_JOBS, NUM_MACHINES))  # Random times

NUM_PARTICLES = 20
MAX_ITER = 50
INERTIA_WEIGHT = 0.5
COGNITIVE_COEFF = 1.5
SOCIAL_COEFF = 1.5

# Fitness Function
def calculate_makespan(schedule, processing_times):
    """Calculate makespan for a given schedule."""
    machine_times = [0] * NUM_MACHINES
    for job, machine in enumerate(schedule):
        machine_times[machine] += processing_times[job][machine]
    return max(machine_times)

# Initialize Particles
def initialize_particles(num_particles, num_jobs, num_machines):
    """Generate particles with random schedules."""
    particles = [np.random.randint(0, num_machines, num_jobs) for _ in range(num_particles)]
    velocities = [np.random.uniform(-1, 1, num_jobs) for _ in range(num_particles)]
    return particles, velocities

# Particle Swarm Optimization
def particle_swarm_optimization():
    particles, velocities = initialize_particles(NUM_PARTICLES, NUM_JOBS, NUM_MACHINES)
    personal_best_positions = particles.copy()
    personal_best_scores = [calculate_makespan(p, PROCESSING_TIMES) for p in particles]
    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    for iteration in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            # Update Velocity
            r1, r2 = np.random.random(), np.random.random()
            velocities[i] = (
                INERTIA_WEIGHT * velocities[i]
                + COGNITIVE_COEFF * r1 * (personal_best_positions[i] - particles[i])
                + SOCIAL_COEFF * r2 * (global_best_position - particles[i])
            )

            # Update Position
            particles[i] = np.clip(np.round(particles[i] + velocities[i]), 0, NUM_MACHINES - 1).astype(int)

            # Evaluate Fitness
            current_score = calculate_makespan(particles[i], PROCESSING_TIMES)

            # Update Personal and Global Best
            if current_score < personal_best_scores[i]:
                personal_best_scores[i] = current_score
                personal_best_positions[i] = particles[i]

            if current_score < global_best_score:
                global_best_score = current_score
                global_best_position = particles[i]

        print(f"Iteration {iteration + 1}: Best Makespan = {global_best_score}")

    return global_best_position, global_best_score

# Run PSO
best_schedule, best_makespan = particle_swarm_optimization()

print("\nProcessing Times:\n", PROCESSING_TIMES)
print("Best Schedule Found:", best_schedule)
print("Best Makespan Achieved:", best_makespan)
