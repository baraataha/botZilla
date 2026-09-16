#  region imports
import cv2 as ai
import mediapipe as mp
from math import sqrt
from os import system

system("cls")
# endregion

#region the dict
left_eye={"upper_right":158,"lower_right":153,"upper_left":160,"lower_left":144,"right_corner":133,"left_corner":33}
right_eye={"upper_right":386,"lower_right":373,"upper_left":385,"lower_left":380,"right_corner":446,"left_corner":398}
shoulder={"right":11,"left":12}
mouth={"right":9,"left":10}

#endregion

# region def
def EAR (d1,d2,w):
    d=(d1+d2)/2
    r=d/w
    return r

def dis (p1,p2):
    x1,y1=p1
    x2,y2=p2
    r=sqrt((x1-x2)**2+(y1-y2)**2)
    return r

def locate (dict,type="face") :
    r=  {}
    
    for name,p in dict.items():
        if type=="face":
          x=face.landmark[p].x*w*2
          y=face.landmark[p].y*h*2
        elif type=="pose":
          x=result.pose_landmarks.landmark[p].x*w*2
          y=result.pose_landmarks.landmark[p].y*h*2            
        r[name]=(x,y)
    
    return r

def mid(p1,p2):
    x1,y1 =p1
    x2,y2 =p2
    x=(x1+x2)/2
    y=(y1+y2)/2
    r=(x,y)
    return r
# endregion

# region setup
x=0
camera = ai.VideoCapture(0)

face_code = mp.solutions.face_mesh
draw_code = mp.solutions.drawing_utils
faces=  face_code.FaceMesh(refine_landmarks= True)

pose_code = mp.solutions.pose
pose =  pose_code.Pose()

# endregion

while True :
    
    #region prepare picture
    msg,frame=camera.read()
    h,w,_=frame.shape
    frame=ai.resize(frame,(w*2,h*2))
    rgb=ai.cvtColor(frame,ai.COLOR_BGR2RGB) 
    # endregion
    
    # region facemesh 
    result=faces.process(rgb)
    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            draw_code.draw_landmarks(frame,
                                    face, 
                                    face_code.FACEMESH_TESSELATION,
                                    draw_code.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=2),
                                    draw_code.DrawingSpec(color=(255,0,0),thickness=1))
    
        
            left_points=locate(left_eye)
            EAR_d_left=[]
            for i in range (3):
                n1,p1=left_points.popitem()
                n2,p2=left_points.popitem()
                EAR_d_left.append(dis(p1,p2))

            EAR_r_left = EAR(EAR_d_left[2],EAR_d_left[1],EAR_d_left[0])            
            right_points=locate(left_eye)
            EAR_d_right=[]
            for i in range (3):
                n1,p1=right_points.popitem()
                n2,p2=right_points.popitem()
                EAR_d_right.append(dis(p1,p2))

            EAR_r_right = EAR(EAR_d_right[2],EAR_d_right[1],EAR_d_right[0])
                
            EAR_r=(EAR_r_left+EAR_r_right)/2
            EAR_r= round(EAR_r*100,0)
            if EAR_r<=16:
                ai.putText(frame,"wakeup",(370,340),ai.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,255),2)
    # endregion
    
    # region pose
    result = pose.process(rgb)    
    if result.pose_landmarks :
        draw_code.draw_landmarks(frame,
                                     result.pose_landmarks,
                                     pose_code.POSE_CONNECTIONS,
                                     draw_code.DrawingSpec(color = (0,255,0),thickness =3 ,circle_radius =4),
                                     draw_code.DrawingSpec(color = (255,0,0),thickness =3))
            
        points = result.pose_landmarks.landmark
        for i  in range(33):
                x = int(points[i].x*w*2)
                y = int(points[i].y*h*2)
                
                ai.putText(
                    frame,
                    str(i),
                    (x,y),
                    ai.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )
        
        shoulder_points = locate(shoulder,"pose")
        mouth_points = locate(mouth,"pose")


        mouth_status=""
        
        mid_shoulder=mid(shoulder_points["right"],shoulder_points["left"])
        mid_mouth=mid(mouth_points["right"],mouth_points["left"])
        
        if mid_mouth[1]>mid_shoulder[1]:
            mouth_status="lower"
        else:
            mouth_status="above"
        
        
        sh_mo_dis=dis(mid_shoulder,mid_mouth)
        
        sh_mo_dis_r=sh_mo_dis/dis(shoulder_points["right"],shoulder_points["left"])*100
        
        system("cls")    
        if (sh_mo_dis_r<20 and mouth_status == "above" )or mouth_status=="lower":
            ai.putText(frame,"fix your pose",(370,340),ai.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,255),2)  
        

                
        ai.imshow("veiwer",frame)
        if ai.waitKey(1) == ord('q') :break
camera.release()
ai.destroyAllWindows()
