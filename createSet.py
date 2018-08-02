import bpy
import math
import random
import os

count = 12

def import_backgrounds():
    directory = os.fsencode('C:\\Users\\Julian\\Desktop\\set\\backgrounds')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        bpy.ops.image.open(filepath='C:\\Users\\Julian\\Desktop\\set\\backgrounds\\'+filename)

def import_object(name):
    file_loc = 'C:\\Users\\Julian\\Desktop\\set\\'+name+'\\'+name+'.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)    

def rotate_cam(amount):
    cam = bpy.context.scene.camera
    cam_loc = cam.location.copy()

    cam_abs = math.sqrt(cam_loc.x*cam_loc.x+cam_loc.y*cam_loc.y)
    alpha = math.atan(cam_loc.y/cam_loc.x)+amount

    if cam_loc.x < 0:
        cam.location.x = -math.cos(alpha)*cam_abs
        cam.location.y = -math.sin(alpha)*cam_abs
    else:
        cam.location.x = math.cos(alpha)*cam_abs
        cam.location.y = math.sin(alpha)*cam_abs

    cam_rot = cam.rotation_euler
    cam_loc = cam.location.copy()
    
    if cam_loc.y < 0:
        cam_rot.z = math.atan(-cam_loc.x/cam_loc.y)
    else:
        cam_rot.z = math.atan(-cam_loc.x/cam_loc.y)+math.pi
        
def reset_cam():
    bpy.context.scene.camera.location.x = -20
    bpy.context.scene.camera.location.y = 0
    bpy.context.scene.camera.location.z = 10
    
    bpy.context.scene.camera.rotation_euler.x = 1.4
    bpy.context.scene.camera.rotation_euler.y = 0
    bpy.context.scene.camera.rotation_euler.z = -math.pi/2

def set_background():
    index = random.randint(1, len(bpy.data.images)-1)
    world = bpy.context.scene.world
    world.active_texture.image = bpy.data.images['background'+str(index)+'.jpg'] 

def main():
    import_backgrounds()
    
    for a in range(100):
        for i in range(count):
            reset_cam()
            rotate_cam(random.random()*math.pi*2)
        
            objects = bpy.context.visible_objects
    
            for obj in objects:
                if obj.type != 'CAMERA' and obj.type != 'LAMP':
                    obj.select = True
                    bpy.ops.object.delete()
        
            set_background()
            import_object('object'+str(i))
        
            bpy.context.scene.render.filepath = 'C:\\Users\\Julian\\Desktop\\set\\output\\'+str(a)+'-'+str(i)+'_0.png'
            bpy.ops.render.render(write_still=True)
        
            rand = random.random()*math.pi/4
            rotate_cam(rand)
            bpy.context.scene.render.filepath = 'C:\\Users\\Julian\\Desktop\\set\\output\\'+str(a)+'-'+str(i)+'_'+str(rand)+'.png'
            bpy.ops.render.render(write_still=True)

main()