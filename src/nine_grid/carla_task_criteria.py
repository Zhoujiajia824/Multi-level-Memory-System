"""Observed task completion, independent of planner and scenario actor commands.

Standard library / Python 3.7. Input actor centres use world XY, CARLA y/yaw
negated. No hidden actor routes, desired controls or predictions are consumed.
"""
import copy
import math


def _section(identity):
    return tuple(identity.split('|')[:2]) if isinstance(identity,str) else None


def _lane(point):
    return '{}|{}|{}'.format(point['road_id'],point.get('section_id',0),point['lane_id'])


def _wrap(value):return (value+math.pi)%(2*math.pi)-math.pi


class TaskCriteria:
    def __init__(self,settings,route_xy=None,physics_hz=50):
        self.hz=physics_hz;self.points=copy.deepcopy(settings['route']['points'])
        self.task=copy.deepcopy(settings['route'].get('task',{}));self.kind=self.task.get('kind',settings.get('family','route'))
        self.route=([list(p) for p in route_xy] if route_xy is not None else [[p['x_m'],-p['y_m']] for p in self.points])
        if len(self.route)!=len(self.points) or len(self.points)<2:raise ValueError('task route geometry and identities must align')
        self.arc=[0.]
        for a,b in zip(self.route,self.route[1:]):self.arc.append(self.arc[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
        self.tick=None;self.progress=0.;self.previous=None;self.records=[];self.checkpoints=[]
        self._next=0;self.wrong_exit=False;self._exit_expected=None;self._exit_mismatch_start=None
        self.overtake={'phase':'not_started','lead_role':self.task.get('lead_role','slow_lead'),'left_tick':None,'passed_tick':None,
            'returned_tick':None,'wrong_side_pass':False,'return_hold_start':None}
        self.roundabout={'entered_tick':None,'left_tick':None,'angular_travel_rad':0.,'previous_angle':None}
        self.turn={'requested':self.task.get('requested_turn'),'entry_tick':None,'exit_tick':None,
            'entry_heading':None,'actual_heading_change_deg':None,'completed':False}
        if self.kind=='signal_junction' and (self.turn['requested'] not in ('left','right','straight')
                or not all(k in self.task for k in ('maneuver_entry','maneuver_exit','maneuver_junction_id'))):
            raise ValueError('signal turn task requires its declared junction entry, exit and maneuver')
        self._build_checkpoints()

    def _build_checkpoints(self):
        # Require actual traversal of each declared non-junction road section.
        # Long sections also have geometric checkpoints every 40 metres. A lane
        # change inside a road section does not invalidate a navigation task.
        selected={0,len(self.points)-1};start=0
        for i in range(1,len(self.points)+1):
            if i<len(self.points) and (_section(_lane(self.points[i])),bool(self.points[i].get('junction')))==(
                    _section(_lane(self.points[start])),bool(self.points[start].get('junction'))):continue
            if not self.points[start].get('junction',False):
                end=i-1
                if self.arc[end]-self.arc[start]>=4.:
                    selected.add(min(end,start+2));selected.add(max(start,end-2))
                    last=self.arc[start]
                    for j in range(start,end+1):
                        if self.arc[j]-last>=40.:selected.add(j);last=self.arc[j]
            start=i
        for index in sorted(selected):
            self.checkpoints.append({'index':index,'route_s_m':self.arc[index],
                'section':list(_section(_lane(self.points[index]))),'position':self.route[index],
                'passed_tick':None})

    def _project(self,position):
        travel=0. if self.previous is None else math.hypot(position[0]-self.previous['position'][0],position[1]-self.previous['position'][1])
        forward=max(5.,travel*2.+2.);best=None
        for i,(a,b) in enumerate(zip(self.route,self.route[1:])):
            if self.arc[i+1]<self.progress-5. or self.arc[i]>self.progress+forward:continue
            dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
            if length<1e-8:continue
            q=max(0.,min(1.,((position[0]-a[0])*dx+(position[1]-a[1])*dy)/length**2))
            ex,ey=position[0]-a[0]-q*dx,position[1]-a[1]-q*dy
            item=(ex*ex+ey*ey,self.arc[i]+q*length,(dx*ey-dy*ex)/length,i)
            if best is None or item[0]<best[0]:best=item
        if best is None:raise ValueError('task observation no longer projects onto local locked route')
        return best[1],best[2],best[3]

    def _record(self,event,tick,**fields):
        self.records.append(dict(event=event,tick=tick,**fields))

    def _navigation(self,tick,ego,index):
        section=_section(ego.get('lane_id'));position=ego['position']
        # Checkpoints must be consumed in order. Position must be near the gate,
        # on its declared road section, and point along the intended direction.
        while self._next<len(self.checkpoints):
            gate=self.checkpoints[self._next];j=gate['index']
            a,b=self.route[max(0,j-1)],self.route[min(len(self.route)-1,j+1)]
            heading=math.atan2(b[1]-a[1],b[0]-a[0])
            if (section!=tuple(gate['section']) or abs(self.progress-gate['route_s_m'])>5.
                    or math.hypot(position[0]-gate['position'][0],position[1]-gate['position'][1])>15.
                    or abs(_wrap(ego['heading']-heading))>math.pi/3):break
            gate['passed_tick']=tick;self._record('route_checkpoint',tick,checkpoint=self._next)
            self._next+=1
        junction=ego.get('is_junction')
        previous_junction=self.previous.get('is_junction') if self.previous is not None else False
        if junction is True and previous_junction is not True:
            in_junction=False
            for point in self.points[index:]:
                if point.get('junction',False):in_junction=True
                elif in_junction:
                    self._exit_expected=_section(_lane(point));break
        if self._exit_expected is not None and junction is False:
            if section==self._exit_expected:
                self._exit_expected=None;self._exit_mismatch_start=None
            elif self._exit_mismatch_start is None:self._exit_mismatch_start=tick
            elif tick-self._exit_mismatch_start>=max(1,round(.5*self.hz)) and not self.wrong_exit:
                self.wrong_exit=True
                self._record('wrong_junction_exit',tick,expected_section=list(self._exit_expected),actual_section=list(section) if section else None)

    def _overtake(self,tick,ego,actors,roles,lateral,index):
        identity=str(roles.get(self.overtake['lead_role'],''))
        lead=next((a for a in actors if str(a['id'])==identity),None)
        if lead is None:return
        c,s=math.cos(lead['heading']),math.sin(lead['heading'])
        dx,dy=ego['position'][0]-lead['position'][0],ego['position'][1]-lead['position'][1]
        longitudinal=dx*c+dy*s;left=-dx*s+dy*c
        half_lengths=(ego['length']+lead['length'])/2.;side_threshold=(ego['width']+lead['width'])/2.+.1
        phase=self.overtake['phase'];aligned=abs(_wrap(ego['heading']-lead['heading']))<.5
        if phase=='not_started' and longitudinal<-half_lengths:
            self.overtake['phase']='following';self._record('following_lead',tick)
        if self.overtake['phase']=='following' and aligned and left>side_threshold and longitudinal<half_lengths+2.:
            self.overtake.update(phase='left_lane',left_tick=tick);self._record('left_lane_for_overtake',tick)
        if aligned and longitudinal>half_lengths+2.:
            if self.overtake['phase']=='left_lane' and left>side_threshold:
                self.overtake.update(phase='passed_left',passed_tick=tick);self._record('passed_lead_on_left',tick)
            elif self.overtake['phase'] in ('not_started','following'):
                self.overtake['wrong_side_pass']=True
        if self.overtake['phase']=='passed_left':
            returned=(ego.get('lane_id')==_lane(self.points[index]) and abs(lateral)<.5
                and longitudinal>half_lengths+2. and aligned)
            if not returned:self.overtake['return_hold_start']=None
            elif self.overtake['return_hold_start'] is None:self.overtake['return_hold_start']=tick
            elif tick-self.overtake['return_hold_start']>=max(1,round(.2*self.hz)):
                self.overtake.update(phase='completed',returned_tick=tick)
                self._record('returned_to_navigation_lane',tick)

    def _circle(self,tick,ego):
        circle=self.task.get('circulation_region_carla_xy_m')
        if not circle:return
        dx,dy=ego['position'][0]-circle[0],ego['position'][1]+circle[1]
        inside=math.hypot(dx,dy)<=circle[2]
        if inside:
            angle=math.atan2(dy,dx)
            if self.roundabout['entered_tick'] is None:
                self.roundabout['entered_tick']=tick;self._record('entered_roundabout',tick)
            old=self.roundabout['previous_angle']
            if old is not None:self.roundabout['angular_travel_rad']+=abs(_wrap(angle-old))
            self.roundabout['previous_angle']=angle
        elif self.roundabout['previous_angle'] is not None:
            self.roundabout['left_tick']=tick;self.roundabout['previous_angle']=None
            self._record('left_roundabout_region',tick)

    def _turn(self,tick,ego):
        if self.turn['completed']:return
        for stage in ('entry','exit'):
            point=self.task['maneuver_'+stage]
            near=(ego.get('is_junction') is False and _section(ego.get('lane_id'))==_section(_lane(point))
                and math.hypot(ego['position'][0]-point['x_m'],ego['position'][1]+point['y_m'])<=6.)
            if not near:continue
            if stage=='entry' and self.turn['entry_tick'] is None:
                self.turn.update(entry_tick=tick,entry_heading=ego['heading'])
                self._record('turn_approach_observed',tick,junction_id=self.task['maneuver_junction_id'])
            elif stage=='exit' and self.turn['entry_tick'] is not None and self.turn['exit_tick'] is None:
                change=math.degrees(_wrap(ego['heading']-self.turn['entry_heading']))
                expected=self.turn['requested']
                completed=(change>30. if expected=='left' else change<-30. if expected=='right' else abs(change)<30.)
                self.turn.update(exit_tick=tick,actual_heading_change_deg=change,completed=completed)
                self._record('turn_exit_observed',tick,heading_change_deg=change,correct_turn=completed)

    def observe(self,tick,ego,actors,actor_roles=None):
        if type(tick) is not int or tick<0 or (self.tick is not None and tick!=self.tick+1):
            raise ValueError('task criteria require every physical tick in order')
        self.progress,lateral,index=self._project(ego['position'])
        self._navigation(tick,ego,index)
        if self.kind=='overtake_return':self._overtake(tick,ego,actors,actor_roles or {},lateral,index)
        if self.kind=='roundabout':self._circle(tick,ego)
        if self.kind=='signal_junction':self._turn(tick,ego)
        self.tick=tick;self.previous=copy.deepcopy(ego)
        return self.report(detailed=False)

    def report(self,detailed=True):
        route_complete=self._next==len(self.checkpoints)
        maneuver=True;reasons=[]
        if self.kind=='overtake_return':
            maneuver=self.overtake['phase']=='completed' and not self.overtake['wrong_side_pass']
            if not maneuver:reasons.append('left_overtake_and_return_not_completed')
        if self.kind=='roundabout':
            maneuver=(self.roundabout['entered_tick'] is not None and self.roundabout['left_tick'] is not None
                and self.roundabout['angular_travel_rad']>=math.pi/4)
            if not maneuver:reasons.append('roundabout_entry_circulation_exit_not_completed')
        if self.kind=='signal_junction':
            maneuver=self.turn['completed']
            if not maneuver:reasons.append('declared_junction_turn_not_observed')
        if not route_complete:reasons.append('ordered_route_checkpoints_not_completed')
        if self.wrong_exit:reasons.append('wrong_junction_exit')
        return {'version':'carla-task-criteria-1.0','kind':self.kind,'completed':route_complete and maneuver and not self.wrong_exit,
            'route_checkpoints_completed':self._next,'route_checkpoints_total':len(self.checkpoints),
            'checkpoints':copy.deepcopy(self.checkpoints) if detailed else None,'wrong_exit':self.wrong_exit,
            'maneuver_completed':maneuver,'overtake':copy.deepcopy(self.overtake) if self.kind=='overtake_return' else None,
            'roundabout':copy.deepcopy(self.roundabout) if self.kind=='roundabout' else None,
            'turn':copy.deepcopy(self.turn) if self.kind=='signal_junction' else None,
            'reasons':reasons,'events':copy.deepcopy(self.records) if detailed else [copy.deepcopy(e) for e in self.records if e['tick']==self.tick],
            'scope':'observed physical states and published ego route; no planner diagnostics or NPC intent'}
