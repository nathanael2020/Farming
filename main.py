import pygame
import sys
import numpy as np
import noise
import random
import math
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

WATER = (60, 140, 255)
DIRT = (228, 183, 133)
PLANT = (82, 179, 38)

GRID_COLOR = (158, 118, 73)
SIZE = 40
CELL_SIZE = 30

# Agent colors
AGENT_TOP = (255, 195, 0)
AGENT_ORIENTATION = (0, 150, 0)

class IsometricGridWorld:
    def __init__(self, size=SIZE, cell_size=CELL_SIZE, iso_offset_x = WINDOW_WIDTH / 2, iso_offset_z = 100):
        self.size = size
        self.iso_offset_x = iso_offset_x
        self.iso_offset_z = iso_offset_z
        self.cell_size = cell_size
        self.grid = {}
        self.agent_position = (10, 0, 10)
        self.agent_direction = (1, 0, 0)  # Initial agent direction (facing right)
        self.new_agent = Agent(self.grid, self.agent_position, self.agent_direction)
        self.rotate_x_angle_degrees = 30
        self.rotate_y_angle_degrees = 45
        self.background_color = (183, 207, 223)
        self.grid_edge_left_color = (201, 147, 88)
        self.grid_edge_right_color = (123, 94, 62)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.plants = []

    # Write the entire gridworld to a file
    def write_grid_to_file(self, gridworld, file_path):
        with open(file_path, 'w') as file:
            json.dump({"gridworld": {k: v.to_dict() for k, v in gridworld.items()}}, file, indent=4)

    def write_grid_row_to_file(self, gridworld, x, file_path):
        row_data = {}
        for z in range(SIZE):
            key = f"{x},0,{z}"
            if key in gridworld:
                row_data[key] = gridworld[key].to_dict()

        with open(file_path, 'w') as file:
            json.dump({"gridworld": row_data}, file, indent=4)

    # Function to load the gridworld data from a JSON file
    def load_gridworld_data(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        updated_gridworld = {}
        self.plants = []

        for key, value in data['gridworld'].items():
            value['x'] = value['x']
            value['y'] = value['y']
            value['z'] = value['z']
            value['terrain_type'] = value['terrain_type'].capitalize()  # Uppercasing the first character
            value['humidity'] = value['humidity']
            value['temperature'] = value['temperature']
            value['nitrogen'] = value['nitrogen']
            value['phosphorus'] = value['phosphorus']
            value['potassium'] = value['potassium']
            updated_gridworld[key] = GridSquare(**value)

            if value['terrain_type'] == 'Plant':
                new_plant = Plant((value['x'], 0, value['z']))
                self.plants.append(new_plant)

        return updated_gridworld

    def generate_grid_from_file(self, filename):

        json_file_path = filename
#        output_file_path = 'gridworld_data.json'
#        gridworld = self.load_gridworld_data(json_file_path)

        self.grid = self.load_gridworld_data(json_file_path)

    #        self.grid = {}

    def generate_grid(self):

        scale = 0.09  # Adjust scale for noise density
        seed = random.randint(0, 20)  # Random seed for generating different terrain
        water_threshold = 0.4  # Threshold for determining water cells

        self.plants = []

        for x in range(SIZE):
            for z in range(SIZE):
                # Generate Perlin noise value for each cell
                noise_val = noise.pnoise2(x * scale, z * scale, base=seed)

                normalize_noise = (noise_val + 1) / 2

                self.grid[f"{x},0,{z}"] = GridSquare(x, 0, z,"Dirt", 50, 20, 5, 3, 2)

                if normalize_noise < water_threshold:
                    self.grid[f"{x},0,{z}"].terrain_type = 'Water'

    def write_grid_to_file(self, filename):

        with open(filename, 'w') as file:
            json.dump({"gridworld": {k: v.to_dict() for k, v in self.grid.items()}}, file, indent=4)

        # grid_location = self.grid[f"{x},0,0"]
        # print(f"Data at grid location {x},0,0: {grid_location.terrain_type}")

    def get_grid_square(self, x, z):
        return self.grid.get((x, 0, z))

    def rotate_x(self, point, angle):
        x, y, z = point
        rad = math.radians(angle)
        y_new = y * math.cos(rad) - z * math.sin(rad)
        z_new = y * math.sin(rad) + z * math.cos(rad)
        return (x, y_new, z_new)

    def rotate_y(self, point, angle):
        x, y, z = point
        rad = math.radians(angle)
        x_new = x * math.cos(rad) + z * math.sin(rad)
        z_new = -x * math.sin(rad) + z * math.cos(rad)
        return (x_new, y, z_new)

    def to_isometric_cube(self, point):
        # Rotate around Y-axis by 45 degrees and then around X-axis by 35.264 degrees
        rotated = self.rotate_y(point, self.rotate_y_angle_degrees)
#        rotated = self.rotate_y(point, 45)
        rotated = self.rotate_x(rotated, self.rotate_x_angle_degrees)
        # Only X and Y coordinates are used in the 2D projection
        return (rotated[0], rotated[1])

    def draw_polygon(self, vertices, color):

        # Convert to isometric view
        iso_vertices = [self.to_isometric_cube(vertex) for vertex in vertices]

#        y_compression = math.sqrt(3) / 2
        y_compression = 1

        iso_vertices = [(x, z * y_compression) for x, z in iso_vertices]

        # Translate to center of the screen
        iso_vertices = [(x + self.iso_offset_x, z + self.iso_offset_z) for x, z in iso_vertices]

#        print(iso_vertices)
        # Draw the polygon
        pygame.draw.polygon(self.screen, color, iso_vertices, 0)

    def draw_cube(self, x, y, z, top_color, right_color, left_color, is_agent=False, draw_3d_left=False, draw_3d_right=False):

        cube_size = self.cell_size

        grid_color = GRID_COLOR
        # Draw top face
        if self.grid[f"{x},0,{z}"].terrain_type == 'Water':
            grid_color = top_color

        x *= cube_size
        y *= cube_size
        z *= -cube_size

        grid_width = max(2, self.cell_size // 50)

        vertices = [
            (x, y, z), (x + cube_size, y, z), (x + cube_size, y + cube_size, z), (x, y + cube_size, z),
            (x, y + cube_size, z - cube_size), (x + cube_size, y + cube_size, z - cube_size), (x + cube_size, y, z - cube_size),
            (x, y, z - cube_size)
        ]

        top_square_vertices = [
            (x + grid_width, y, z - grid_width), (x + cube_size - grid_width, y, z - grid_width), (x + cube_size - grid_width, y, z - cube_size + grid_width), (x + grid_width, y, z - cube_size + grid_width)
        ]

        if is_agent:

            if self.agent_direction == (0, 0, 1):
                orientation_vertices = [
                    (x + grid_width, y, z - cube_size * .75), (x + cube_size - grid_width, y, z - cube_size * .75), (x + cube_size - grid_width, y, z - cube_size + grid_width),
                    (x + grid_width, y, z - cube_size + grid_width)
                ]
            elif self.agent_direction == (0, 0, -1):
                orientation_vertices = [
                    (x + grid_width, y, z - cube_size * .25), (x + cube_size - grid_width, y, z - cube_size * .25), (x + cube_size - grid_width, y, z - grid_width),
                    (x + grid_width, y, z - grid_width)
                ]
            elif self.agent_direction == (1, 0, 0):
                orientation_vertices = [
                    (x + cube_size * 0.75, y, z - grid_width), (x + cube_size * 0.75, y, z - cube_size + grid_width), (x + cube_size - grid_width, y, z - cube_size + grid_width),
                    (x + cube_size - grid_width, y, z - grid_width)
                ]
            elif self.agent_direction == (-1, 0, 0):
                orientation_vertices = [
                    (x + grid_width, y, z - grid_width), (x + grid_width, y, z - cube_size + grid_width), (x + cube_size * 0.25, y, z - cube_size + grid_width),
                    (x + cube_size * 0.25, y, z - grid_width)
                ]

            orientation_face = [orientation_vertices[0], orientation_vertices[1], orientation_vertices[2],
                                orientation_vertices[3]]

        # Define faces of the cube
        top_face = [vertices[0], vertices[1], vertices[6], vertices[7]]
        top_square_face = [top_square_vertices[0], top_square_vertices[1], top_square_vertices[2], top_square_vertices[3]]
        left_face = [vertices[4], vertices[5], vertices[6], vertices[7]]
        right_face = [vertices[1], vertices[2], vertices[5], vertices[6]]

        self.draw_polygon(top_face, grid_color)
        self.draw_polygon(top_square_face, top_color)

        if is_agent:
            self.draw_polygon(orientation_face, pygame.Color(AGENT_ORIENTATION))

        if draw_3d_right:
            # Draw right face
            self.draw_polygon(right_face, right_color)
        if draw_3d_left:
            # Draw left face
            self.draw_polygon(left_face, left_color)

    def to_isometric(self, x, z):

        cell_size = self.cell_size

        # Define the angle for isometric projection
        theta = math.radians(self.rotate_y_angle_degrees)  # 45 degrees in radians

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Apply rotate and scale transformation
        rotated_vector = np.dot(rotation_matrix, np.array([x, z]))

        # Apply scaling to the y-coordinate
        z_scale = math.cos(math.radians(35.264))  # Vertical compression factor
        iso_x = rotated_vector[0]
        iso_z = rotated_vector[1] * z_scale

        # Scale both coordinates with cell size and adjust for screen positioning
        iso_x = iso_x * cell_size + self.iso_offset_x
        iso_z = iso_z * cell_size + self.iso_offset_z

        return iso_x, iso_z

    def move_agent(self, direction):
        x, y, z = self.agent_position
        dx, dy, dz = self.agent_direction
        new_x = x
        new_y = y
        new_z = z

        if direction == 'forward':
            new_x = x + dx
            new_y = y
            new_z = z + dz
        elif direction == 'backward':
            new_x = x - dx
            new_y = y
            new_z = z - dz
        elif direction == 'up':
            new_x = x
            new_y = y - 1
            new_z = z
        elif direction == 'down':
            new_x = x
            new_y = y + 1
            new_z = z
        elif direction == 'rotate_left':
            new_direction = (dz, 0, -dx)  # Rotate 90 degrees to the right
            self.agent_direction = new_direction
            self.new_agent.orientation = new_direction
        elif direction == 'rotate_right':
            new_direction = (-dz, 0, dx)  # Rotate 90 degrees to the right
            self.agent_direction = new_direction
            self.new_agent.orientation = new_direction

        if 0 <= new_x < self.size and 0 <= new_z < self.size:
            self.agent_position = (new_x, new_y, new_z)

        self.new_agent.location = self.agent_position
        self.new_agent.orientation = self.agent_direction

    def display_square_info(self, screen, square, position=(10, 10)):

        font = pygame.font.SysFont('Arial', 20)

        attributes = [
            f'X: {square.x}',
            f'Z: {square.z}',
            f'Terrain Type: {square.get_terrain_type()}',
            f'Humidity: {square.humidity}',
            f'Temperature: {square.temperature}',
            f'Nitrogen: {square.nitrogen}',
            f'Phosphorus: {square.phosphorus}',
            f'Potassium: {square.potassium}'
        ]

        pygame.draw.rect(screen, (255, 255, 255), (position[0], position[1], 200, 200))

        for i, attribute in enumerate(attributes):
            text = font.render(attribute, True, (0, 0, 0))
            screen.blit(text, (position[0], position[1] + i * 20))

    def make_it_rain(self):
        for x in range(self.size):
            for z in range(self.size):
                self.grid[f"{x},0,{z}"].humidity += random.randint(200, 1000)

    def interact(self):
        x, y, z = self.agent_position
        dx, dy, dz = self.agent_direction

        # Calculate the position of the cell in front of the agent
        front_x = x + dx
        front_z = z + dz

        # Check if the cell is within the bounds of the grid
        if 0 <= front_x < self.size and 0 <= front_z < self.size:

            grid_square = self.grid[f"{front_x},0,{front_z}"]

            if grid_square is not None and grid_square.terrain_type == 'Water':
                grid_square.terrain_type = 'Dirt'
                new_plant = Plant((front_x, 0, front_z))
                self.plants.append(new_plant)
                self.redraw_grid(new_plant.location)
            elif grid_square is not None and grid_square.terrain_type != 'Plant':
                grid_square.terrain_type = 'Plant'
                new_plant = Plant((front_x, 0, front_z))
                self.plants.append(new_plant)
                self.redraw_grid(new_plant.location)

            # # Toggle the state of the cell in front of the agent
            # self.get_grid_square(front_x, front_z).set_terrain_type("Plant")
            # new_plant = Plant((front_x, 0, front_z))
            # self.plants.append(new_plant)
            # self.redraw_grid(new_plant.location)

    # def check_neighbors(self, x, z):
    #
    #     neighbors = []
    #     planted = False
    #
    #     for i in range(-1, 2):
    #         for j in range(-1, 2):
    #             if 0 <= x + i < self.size and 0 <= z + j < self.size:
    #                 if i == 0 and j == 0:
    #                     continue
    #                 else:
    #                     if self.grid[x + i, 0, z + j] == 50:
    #                         planted = True
    #                         neighbors.append((x + i, 0, z + j))
    #
    #     return planted

    def run(self):

        self.screen.fill(self.background_color)
#        self.draw_grid_lines((113, 177, 223))

        self.generate_grid()

        self.draw_grid()

        while True:

            self.keyboard_controls()

#            self.clock.tick(60)
            fps = self.clock.get_fps()
            pygame.display.set_caption(f'FPS: {fps:.2f}')

            self.update_world()

            self.display_square_info(self.screen, self.grid[f"{self.agent_position[0]},0,{self.agent_position[2]}"], position=(10, 10))

            pygame.display.flip()

    def keyboard_controls(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.move_agent('forward')
                    self.redraw_grid(self.agent_position)
                elif event.key == pygame.K_s:
                    self.move_agent('backward')
                    self.redraw_grid(self.agent_position)
                elif event.key == pygame.K_a:
                    self.move_agent('rotate_left')
                    self.redraw_grid(self.agent_position)
                elif event.key == pygame.K_d:
                    self.move_agent('rotate_right')
                    self.redraw_grid(self.agent_position)
                elif event.key == pygame.K_SPACE:
                    self.interact()
                    self.redraw_grid(self.agent_position)
                elif event.key == pygame.K_r:
                    self.make_it_rain()
                elif event.key == pygame.K_UP:
                    self.iso_offset_z -= 50
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_DOWN:
                    self.iso_offset_z += 50
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_LEFT:
                    self.iso_offset_x -= 50
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_RIGHT:
                    self.iso_offset_x += 50
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_o:
                    self.rotate_x_angle_degrees = min(70, self.rotate_x_angle_degrees + 5)
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_p:
                    self.rotate_x_angle_degrees = max(20, self.rotate_x_angle_degrees - 5)
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_k:
                    self.rotate_y_angle_degrees = min(70, self.rotate_y_angle_degrees + 5)
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_l:
                    self.rotate_y_angle_degrees = max(20, self.rotate_y_angle_degrees - 5)
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_z:
                    self.cell_size = max(5, self.cell_size * .9)
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_x:
                    self.cell_size = min(200, self.cell_size * 1.1)
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_y:
                    self.generate_grid_from_file('gridworld_data2.json')
                    self.screen.fill(self.background_color)
                    self.draw_grid()
                elif event.key == pygame.K_t:
                    self.write_grid_to_file('gridworld_data2.json')
                elif event.key == pygame.K_u:
                    self.generate_grid()
                    self.draw_grid()
                elif event.key == pygame.K_b:
                    self.new_agent.sense()
                elif event.key == pygame.K_g:
                    self.new_agent.set_world_model('B')
                    self.new_agent.run(100000)
                    print(f"Agent location: {self.new_agent.location}")
                    print(f"Agent orientation: {self.new_agent.orientation}")
                    self.agent_position = self.new_agent.location
                    self.agent_direction = self.new_agent.orientation
                    self.draw_grid()

    def update_world(self):

        self.grow_plants()
        self.agent_position = self.new_agent.location
        self.agent_direction = self.new_agent.orientation

        for plant in self.plants:

            nearby_water = False

            for x in range(max(plant.location[0] - 2, 0), min(plant.location[0] + 3, self.size)):
                for z in range(max(plant.location[2] - 2, 0), min(plant.location[2] + 3, self.size)):
                    if self.grid[f"{x},0,{z}"].terrain_type == 'Water':
                        nearby_water = True

            grid_square = self.grid[f"{plant.location[0]},0,{plant.location[2]}"]

            if nearby_water:
                for x in range(max(plant.location[0] - 1, 0), min(plant.location[0] + 2, self.size)):
                    for z in range(max(plant.location[2] - 1, 0), min(plant.location[2] + 2, self.size)):
                        self.grid[f"{x},0,{z}"].humidity += 1
                        if self.grid[f"{x},0,{z}"].humidity > 1000:
                            self.grid[f"{x},0,{z}"].humidity = 1000
            else:
                grid_square.humidity -= 2
                if grid_square.humidity < 0:
                    grid_square.humidity = 0

            plant.increase_age(int(max(1, 10 - int(grid_square.humidity) / 10)))

            if plant.status == 'Dead':
                grid_square.terrain_type = 'Dirt'
                self.redraw_grid(plant.location)
                self.plants.remove(plant)





    def grow_plants(self):

        for plant in self.plants:

            # Coordinates for the adjacent squares in the cardinal directions
            adjacent_coordinates = [
                (plant.location[0] - 1, plant.location[2]),  # West
                (plant.location[0] + 1, plant.location[2]),  # East
                (plant.location[0], plant.location[2] - 1),  # North
                (plant.location[0], plant.location[2] + 1)  # South
            ]

            # for x in range(max(plant.location[0] - 1, 0), min(plant.location[0] + 2, self.size)):
            #     for z in range(max(plant.location[2] - 1, 0), min(plant.location[2] + 2, self.size)):

            for x, z in adjacent_coordinates:

                if x < 0 or x >= self.size or z < 0 or z >= self.size:
                    continue
                else:
                    grid_square = self.grid[f"{x},0,{z}"]
                    # grid_square.terrain_type = 'Plant'

                    if grid_square is not None and grid_square.terrain_type != 'Plant' and grid_square.humidity > 10 and grid_square.terrain_type != 'Water':
                        rand = random.randint(0, 1000)
                        if rand <= 10:
                            grid_square.terrain_type = 'Plant'
                            new_plant = Plant((x, 0, z))
                            self.plants.append(new_plant)
                            self.redraw_grid(new_plant.location)

                        # if self.get_grid_square(x, z).get_terrain_type != 'Plant':
                        #     rand = random.randint(0, 1000)
                        #     if rand <= 1:
                        #         self.get_grid_square(x, z).set_terrain_type('Plant')
                        #         self.plants.append(Plant((x, 0, z)))
                        #         self.redraw_grid((x, 0, z))

    def draw_grid(self):
        for x in range(self.size):
            for z in range(self.size):

                top_color = self.grid[f"{x},0,{z}"].get_terrain_color_rgb()

                if (x, 0, z) == self.agent_position:
                    self.draw_cube(x, 0, z, pygame.Color(AGENT_TOP), pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=True, draw_3d_left=False, draw_3d_right=False)
                else:
                    if x == self.size - 1 and z == self.size - 1:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=False, draw_3d_left=True, draw_3d_right=True)
                    elif x == self.size - 1:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color),
                                       is_agent=False, draw_3d_left=False, draw_3d_right=True)
                    elif z == self.size - 1:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=False, draw_3d_left=True, draw_3d_right=False)
                    else:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=False, draw_3d_left=False, draw_3d_right=False)

        print("Grid Drawn")

    def redraw_grid(self, position):
        for x in range(max(0,position[0] - 5), min(self.size,position[0] + 5)):
            for z in range(max(0, position[2] - 5), min(self.size, position[2] + 5)):

                top_color = self.grid[f"{x},0,{z}"].get_terrain_color_rgb()

                if (x, 0, z) == self.agent_position:
                    self.draw_cube(x, 0, z, pygame.Color(AGENT_TOP), pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=True, draw_3d_left=False, draw_3d_right=False)
                else:
                    if x == self.size - 1 and z == self.size - 1:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=False, draw_3d_left=True, draw_3d_right=True)
                    elif x == self.size - 1:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color),
                                       is_agent=False, draw_3d_left=False, draw_3d_right=True)
                    elif z == self.size - 1:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=False, draw_3d_left=True, draw_3d_right=False)
                    else:
                        self.draw_cube(x, 0, z, top_color, pygame.Color(self.grid_edge_right_color), pygame.Color(self.grid_edge_left_color), is_agent=False, draw_3d_left=False, draw_3d_right=False)

class Agent:
    def __init__(self, grid, location, orientation):
        self.location = location
        self.orientation = orientation
        self.world_model = {}
        self.grid = grid
#        self.set_world_model()
        self.size = SIZE

    def create_character_image(character):
        # Create a blank image with white background
        image = Image.new('L', (30, 30), 0)
        draw = ImageDraw.Draw(image)

        # Define the font and size
        font = ImageFont.truetype('/Library/Fonts/BenchNine-Bold.ttf', 26)

        # Draw the character
        draw.text((5, 0), character, fill=1, font=font)

        # Convert to numpy array and binarize
        data = np.array(image)
        data = (data > 0).astype(int)
        return data

    def load_grid_from_json(self, file_path):
        """
        Load grid data from a JSON file into a Python array.

        :param file_path: Path to the JSON file.
        :return: Array representing the grid data.
        """
        try:
            with open(file_path, 'r') as file:
                grid_data = json.load(file)
            return grid_data
        except FileNotFoundError:
            print("File not found. Please check the file path.")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON. Please check the file contents.")
            return None


    def create_character_image(self, character):
        # Create a blank image with white background
        image = Image.new('L', (40, 40), 0)
        draw = ImageDraw.Draw(image)

        # Define the font and size
        font = ImageFont.truetype('/Library/Fonts/BenchNine-Bold.ttf', 36)

        # Draw the character
        draw.text((5, 0), character, fill=1, font=font)

        image = ImageOps.mirror(image)
        # Convert to numpy array and binarize
        data = np.array(image)
        data = (data > 0).astype(int)
        return data

    def set_world_model(self, c):

        # Characters to create images for
        characters = c

        character_images = {char: self.create_character_image(char) for char in characters}

        # Prepare the input and output data
        X = np.array([character_images[char] for char in characters])
        y = np.array(list(range(len(characters))))  # Numeric labels

        # Regenerate the character images and print their array representations
        character_images = {char: self.create_character_image(char) for char in characters}

        # Convert each character image to a string representation of a 2D array and print them
        array_strings = {char: '\n'.join(' '.join(str(cell) for cell in row) for row in image)
                         for char, image in character_images.items()}

        # Display the array representation for each character
        for char in characters:  # Displaying all characters
            print(f"Character: {char}\n{array_strings[char]}\n")

        # Convert binary grid to DIRT/WATER representation for each character
        grid_data = {char: [["Dirt" if cell == 0 else "Water" for cell in row] for row in image]
                      for char, image in character_images.items()}

        # Update world model based on the character grid
        for char, grid in grid_data.items():
            for x, row in enumerate(grid):
                for z, cell in enumerate(row):
                    self.world_model[f"{x},0,{z}"] = cell

        # for x in range(SIZE):
        #     for z in range(SIZE):
        #         if SIZE // 2 - 10 <= x <= SIZE // 2 + 10 and SIZE // 2 - 10 <= z <= SIZE // 2 + 10:
        #             self.world_model[f"{x},0,{z}"] = "Water"
        #         else:
        #             self.world_model[f"{x},0,{z}"] = "Dirt"

    def sense(self):
        sensed_data = self.get_sensing_data()
        self.update_world_model(sensed_data)

    def get_sensing_data(self):

        grid_center = tuple(v1 + (2 * v2) for v1, v2 in zip(self.location, self.orientation))

        print(f"Grid center: {grid_center}")

        # Coordinates for the 5x5 subgrid in front of the agent
        subgrid_coordinates = []

        for x in range(grid_center[0] - 2, grid_center[0] + 3):
            for z in range(grid_center[2] - 2, grid_center[2] + 3):
                subgrid_coordinates.append((x, 0, z))

        # Get the terrain type for each cell in the subgrid
        sensed_data = {}
        for coordinate in subgrid_coordinates:
            if coordinate[0] < 0 or coordinate[0] >= SIZE or coordinate[2] < 0 or coordinate[2] >= SIZE:
                sensed_data[coordinate] = 'Outside'
            else:
                sensed_data[coordinate] = self.grid[f"{coordinate[0]},{coordinate[1]},{coordinate[2]}"].get_terrain_type()

            print(f"Sensed data at {coordinate}: {sensed_data[coordinate]}")

        return sensed_data

    def interact(self):
        x, y, z = self.location
        dx, dy, dz = self.orientation

        # Calculate the position of the cell in front of the agent
        front_x = x + dx
        front_z = z + dz

        # Check if the cell is within the bounds of the grid
        if 0 <= front_x < self.size and 0 <= front_z < self.size:

            grid_square = self.grid[f"{front_x},0,{front_z}"]

            if grid_square is not None and grid_square.terrain_type == 'Water' and self.world_model[f"{front_x},0,{front_z}"] == 'Dirt':
                grid_square.terrain_type = 'Dirt'
            elif grid_square is not None and grid_square.terrain_type == 'Dirt' and self.world_model[f"{front_x},0,{front_z}"] == 'Water':
                grid_square.terrain_type = 'Water'

    def run(self, iterations):

        for i in range(iterations):
            rand = random.randint(0, 4)
            if rand == 0:
                self.orientation = (-1, 0, 0)
            elif rand == 1:
                self.orientation = (1, 0, 0)
            elif rand == 2:
                self.orientation = (0, 0, -1)
            elif rand == 3:
                self.orientation = (0, 0, 1)
            self.location = (self.location[0] + self.orientation[0], self.location[1] + self.orientation[1], self.location[2] + self.orientation[2])
            if self.location[0] < 0 or self.location[0] >= SIZE or self.location[2] < 0 or self.location[2] >= SIZE:
                self.orientation = (-self.orientation[0], 0, -self.orientation[2])
                self.location = (self.location[0] + self.orientation[0], self.location[1] + self.orientation[1], self.location[2] + self.orientation[2])
            self.interact()
            

    def update_world_model(self, sensor_data):

        for sensor in sensor_data:
            self.world_model[sensor] = sensor_data[sensor]





class Plant:
    def __init__(self, location):
        self.location = location
        self.growth = 0
        self.lifetime = random.randint(100, 3000)
        self.age = 0
        self.status = 'Alive'

    def increase_age(self, age_increase):
        self.age += age_increase
        if self.age >= self.lifetime:
            self.status = 'Dead'


class GridSquare:
    def __init__(self, x, y, z, terrain_type, humidity=0, temperature=0, nitrogen=0, phosphorus=0, potassium=0):
        self.x = x
        self.y = y
        self.z = z
        self.terrain_type = terrain_type
        self.humidity = humidity
        self.temperature = temperature
        self.nitrogen = nitrogen
        self.phosphorus = phosphorus
        self.potassium = potassium

        self.randomize_attributes()

    def get_terrain_type(self):
        return self.terrain_type

    def set_terrain_type(self, terrain_type):
        self.terrain_type = terrain_type

    def get_terrain_color_rgb(self):
        terrain_colors = {
            "Water": WATER,
            "Dirt": (228, 183, 133),
            "Plant": (82, 179, 38)
        }

        return terrain_colors.get(self.terrain_type)

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "terrain_type": self.terrain_type,
            "humidity": self.humidity,
            "temperature": self.temperature,
            "nitrogen": self.nitrogen,
            "phosphorus": self.phosphorus,
            "potassium": self.potassium
        }

    def randomize_attributes(self):
        self.humidity = random.randint(0, 100)
        self.temperature = random.randint(0, 100)
        self.nitrogen = random.randint(0, 100)
        self.phosphorus = random.randint(0, 100)
        self.potassium = random.randint(0, 100)


# Initialize pygame
pygame.init()
grid_world = IsometricGridWorld()
grid_world.run()  # This will run the grid world simulation
