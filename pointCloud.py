import math
import numpy as np
import PIL as Image
    
cloud = []
size = 10   
width = 0
height = 0

planeR = []
planeBot = []
planeF = []

# create point cloud boundaries

def create_planes(base):
    norm = -normalize(base)
    planeF.extend([norm, np.multiply(norm, base)])

    origin = get_coords(width/2, 0, base)
    norm = -normalize(np.subtract(origin, base))
    planeBot.extend([norm, np.multiply(norm, origin)])

    origin = get_coords(0, height/2, base)
    norm = -normalize(np.subtract(origin, base))
    planeR.extend([norm, np.multiply(norm, origin)])

# project image coords to mapped frame

def get_coords(x, y, base):
    ang_z = math.atan(base[1]/base[0])
    ang_y = math.atan(base[2]/base[0])
        
    rot_z = np.array([[math.cos(ang_z), -math.sin(ang_z), 0], [math.sin(ang_z), math.cos(ang_z), 0], [0, 0, 1]])
    rot_y = np.array([[math.cos(ang_y), 0, math.sin(ang_y)], [0, 1, 0], [-math.sin(ang_y), 0, math.cos(ang_y)]])

    offset = np.array([0, x, y])
    base_new = np.add(base, np.multiply(np.multiply(offset, rot_z), rot_y))

    return base_new
    
# create normal line from point

def getLine(x, y, base_vec):
    local_base = get_coords(x, y, base_vec)
    local_dir = normalize(base_vec)
    return (local_base, local_dir)

# normalizing the ionput vector to length 1

def normalize(vec):
    return np.divide(vec, np.absolute(vec))

# returns a tuple of distances to the cloud boundaries

def get_all_distances(base):
    dist_F = get_dist(planeF, base) 
    dist_Bot = get_dist(planeBot, base)
    dist_R = get_dist(planeR, base)

    return [dist_F, dist_Bot, dist_R]

# returns a tuple with the relative incrementation for each axis

def get_ray_incrementation(line):
    dist_F = get_dist([planeF[0], np.multiply(planeF[0], line[0])], np.add(line[0], line[1]))
    dist_Bot = get_dist([planeBot[0], np.multiply(planeBot[0], line[0])], np.add(line[0], line[1]))
    dist_R = get_dist([planeR[0], np.multiply(planeR[0], line[0])], np.add(line[0], line[1]))

    return (dist_F, dist_Bot, dist_R)

# calculate distance from camera to plane/pixel ray to parallel plane

def get_dist(plane, point):
    return (plane[1]-np.multiply(plane[0], point))/(np.absolute(plane[0])**2)

# returns true if the vectors point in the same hemisphere

def is_same_dir(vec1, vec2):
    return math.acos(np.multiply(vec1, vec2)/(np.absolute(vec1)*np.absolute(vec2))) <= math.pi/2
    
# main executing method;
# used to create the point cloud and iterate through the frames

def create_cloud(self, pics):

    coords_base = np.array(pics[0][0], np.float)
    self.width += pics[0][1].width
    self.height += pics[0][1].height

    create_planes(coords_base)
    
    for i in range(len(pics)):
        
        fact = pics[i][1].width/size

        base_vec = np.array(pics[i][0], np.float)
        img = pics[i][1].load()

        if i==0:
            for x in range(size):
                cloud.append([])
                for y in range(size*height/width):
                    cloud[x].append([])
                    for z in range(size):
                        cloud[x][y].append([0.0, img[x*fact][y*fact]])
        else:
            for x in range(size):
                for y in range(size*height/width):
                    line = getLine(x, y, coords_base)
                    dists = get_all_distances(base_vec)
                    incs = get_ray_incrementation(line)

                    if dists[1] < 0:
                        if is_same_dir(planeBot[0], line[1]):
                            dists[1] *= -1
                        else:
                            dists[1] = height + dists[1]
                    else:
                        dists[1] *= -1

                    if dists[0] < 0:
                        if is_same_dir(planeF[0], line[1]):
                            dists[0] *= -1
                        else:
                            dists[0] = size+dists[0]
                    else:
                        dists[0] *= -1

                    if dists[2] < 0:
                        if is_same_dir(planeR[0], line[1]):
                            dists[2] *= -1
                        else:
                            dists[2] = width+dists[2]
                    else:
                        dists[2] *= -1

                    max_dist = 0

                    if dists[0] < 0 and -dists[0]/incs[0] > max_dist:
                        max_dist = -dists[0]/incs[0]
                                        
                    if dists[1] < 0 and -dists[1]/incs[1] > max_dist:
                        max_dist = -dists[1]/incs[1]

                    if dists[2] < 0 and -dists[2]/incs[2] > max_dist:
                        max_dist = -dists[2]/incs[2]

                    min_steps = math.inf

                    if size-dists[0]/incs[0]-max_dist < min_steps:
                        min_steps = size-dists[0]/incs[0]-max_dist

                    if height-dists[1]/incs[1]-max_dist < min_steps:
                        min_steps = height-dists[1]/incs[1]-max_dist

                    if width-dists[2]/incs[2]-max_dist < min_steps:
                        min_steps = width-dists[2]/incs[2]-max_dist

                    add_x = max_dist+dists[2]/incs[2]
                    add_y = max_dist+dists[1]/incs[1]
                    add_z = max_dist+dists[0]/incs[0]

                    for z in range(min_steps):
                        if cloud[(z+add_x)*incs[2]][(z+add_y)*incs[1]][(z+add_z)*incs[0]] == img[x][y]:
                            cloud[(z+add_x)*incs[2]][(z+add_y)*incs[1]][(z+add_z)*incs[0]][0] += 1

