import copy
import xml.etree.ElementTree as xml
import h5py
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from torch.utils.data import Dataset
import numpy as np
from datasets.utils import LL2XYProjector, Point, get_type, get_subtype, get_x_y_lists, \
    get_relation_members


class SITPDataset(Dataset):
    def __init__(self, dset_path, split_name="train",evaluation=False,args=None):
        self.args =args
        self.data_root = dset_path
        self.split_name = split_name
        self.pred_horizon = args.pred_len // 2
        self.hist_horizon = args.hist_len // 2
        self.num_agent_types = 2
        self.predict_yaw = True
        self.map_attr = 7
        self.k_attr = 8

        dataset = h5py.File(os.path.join(self.data_root, split_name + '_dataset.hdf5'), 'r')
        self.dset_len = len(dataset["agents_trajectories"])

        roads_fnames = glob.glob(os.path.join(self.data_root, "maps", "*.osm"))
        self.max_num_pts_per_road_seg = 0
        self.max_num_road_segs = 0
        self.roads = {}
        for osm_fname in roads_fnames:
            road_info = self.get_map_lanes(osm_fname, 0., 0.)
            if len(road_info) > self.max_num_road_segs:
                self.max_num_road_segs = len(road_info)
            key_fname = osm_fname.split("/")[-1]
            self.roads[key_fname] = road_info

        self.evaluation = evaluation

        if not evaluation:
            self.num_others = 8  # train with 8 agents
        else:
            self.num_others = 40  # evaluate with 40 agents.

    def get_map_lanes(self, filename, lat_origin, lon_origin):
        projector = LL2XYProjector(lat_origin, lon_origin)

        e = xml.parse(filename).getroot()

        point_dict = dict()
        for node in e.findall("node"):
            point = Point()
            point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
            point_dict[int(node.get('id'))] = point

        unknown_linestring_types = list()
        road_lines = []

        road_lines_dict = {}
        exlusion_ids = []

        min_length = 40
        for way in e.findall('way'):
            way_type = get_type(way)

            if way_type is None:
                raise RuntimeError("Linestring type must be specified")
            elif way_type == "curbstone":
                mark_type = np.array([1.0, 0.0, 0.0, 1.0])
            elif way_type == "line_thin":
                way_subtype = get_subtype(way)
                if way_subtype == "dashed":
                    mark_type = np.array([1.0, 1.0, 0.0, 1.0])
                else:
                    mark_type = np.array([0.0, 1.0, 0.0, 1.0])
            elif way_type == "line_thick":
                way_subtype = get_subtype(way)
                if way_subtype == "dashed":
                    mark_type = np.array([1.0, 1.0, 0.0, 1.0])
                else:
                    mark_type = np.array([0.0, 1.0, 0.0, 1.0])
            elif way_type == "pedestrian_marking":
                mark_type = np.array([0.0, 0.0, 1.0, 1.0])
            elif way_type == "bike_marking":
                mark_type = np.array([0.0, 0.0, 1.0, 1.0])
            elif way_type == "stop_line":
                mark_type = np.array([1.0, 0.0, 1.0, 1.0])
            elif way_type == "virtual":
                # mark_type = np.array([1.0, 1.0, 0.0, 1.0])
                exlusion_ids.append(way.get("id"))
                continue
            elif way_type == "road_border":
                mark_type = np.array([1.0, 1.0, 1.0, 1.0])
            elif way_type == "guard_rail":
                mark_type = np.array([1.0, 1.0, 1.0, 1.0])
            elif way_type == "traffic_sign":
                exlusion_ids.append(way.get("id"))
                continue
            else:
                if way_type not in unknown_linestring_types:
                    unknown_linestring_types.append(way_type)
                continue

            x_list, y_list = get_x_y_lists(way, point_dict)
            if len(x_list) < min_length:
                x_list = np.linspace(x_list[0], x_list[-1], min_length).tolist()
                y_list = np.linspace(y_list[0], y_list[-1], min_length).tolist()

            lane_pts = np.array([x_list, y_list]).transpose() # [40,2]
            mark_type = np.zeros((len(lane_pts), 4)) + mark_type # [40,4f]
            if len(x_list) > self.max_num_pts_per_road_seg:
                self.max_num_pts_per_road_seg = len(x_list)

            lane_pts = np.concatenate((lane_pts, mark_type), axis=1) # [40,6]
            road_lines.append(lane_pts)
            road_lines_dict[way.get("id")] = lane_pts

        new_roads = np.zeros((160, self.max_num_pts_per_road_seg, 6))  #160LN,40waypoint,6F empirically found max num roads is 157.
        for i in range(len(road_lines)):
            new_roads[i, :len(road_lines[i])] = road_lines[i]

        used_keys_all = []
        num_relations = len(e.findall('relation'))
        relation_lanes = np.zeros((num_relations + len(road_lines), 80, 8))
        counter = 0
        for rel in e.findall('relation'):
            rel_lane, used_keys = get_relation_members(rel, road_lines_dict, exlusion_ids)
            if rel_lane is None:
                continue
            used_keys_all += used_keys
            new_lanes = np.array(rel_lane).reshape((-1, 8))
            relation_lanes[counter] = new_lanes
            counter += 1

        # delete all used keys
        used_keys_all = np.unique(used_keys_all)
        for used_key in used_keys_all:
            del road_lines_dict[used_key]

        # add non-used keys
        for k in road_lines_dict.keys():
            relation_lanes[counter, :40, :5] = road_lines_dict[k][:, :5]  # rest of state (position (2), and type(3)).
            relation_lanes[counter, :40, 5:7] = -1.0  # no left-right relationship
            relation_lanes[counter, :40, 7] = road_lines_dict[k][:, -1]  # mask
            counter += 1

        return relation_lanes[relation_lanes[:, :, -1].sum(1) > 0]

    def split_input_output_normalize(self, agents_data, meta_data, agent_types):
        if self.evaluation: 
            in_horizon = self.hist_horizon * 2
            out_horizon = self.pred_horizon * 2
        else:
            in_horizon = self.hist_horizon
            out_horizon = self.pred_horizon
        total_horizon = in_horizon + out_horizon
        agents_data = agents_data[:,:total_horizon]

        agent_masks = np.expand_dims(agents_data[:, :, 0] != -1, axis=-1).astype(np.float32)  

        agents_data[:, :, :2] -= np.array([[meta_data[0], meta_data[1]]])  # Normalize with translation only，
        agents_data = np.nan_to_num(agents_data, nan=-1.0)  # pedestrians have nans instead of yaw and size
        agents_data = np.concatenate([agents_data, agent_masks], axis=-1) # [50,40hist+pred,8]

        dists = euclidean_distances(agents_data[:, in_horizon - 1, :2], agents_data[:, in_horizon - 1, :2]) 
        agent_masks[agent_masks == 0] = np.nan
        dists *= agent_masks[:, in_horizon - 1]
        dists *= agent_masks[:, in_horizon - 1].transpose()
        ego_idx = np.random.randint(0, int(np.nansum(agent_masks[:, in_horizon - 1]))) 
        closest_agents = np.argsort(dists[ego_idx])
        agents_data = agents_data[closest_agents[:self.num_others + 1]] 
        agent_types = agent_types[closest_agents[:self.num_others + 1]] 

        agents_in = agents_data[1:(self.num_others + 1), :in_horizon] 
        agents_out = agents_data[1:(self.num_others + 1), in_horizon:, [0, 1, 4, 7]]  
        ego_in = agents_data[0, :in_horizon]
        ego_out = agents_data[0, in_horizon:] 
        ego_out = ego_out[:, [0, 1, 4, 7]]  

        return ego_in, ego_out, agents_in, agents_out, agent_types

    def copy_agent_roads_across_agents(self, agents_in, roads):
        new_roads = np.zeros((self.num_others + 1, *roads.shape)) 
        new_roads[0] = roads  # ego
        for n in range(self.num_others):
            if agents_in[n, -1, -1]:
                new_roads[n + 1] = roads
        return new_roads

    def make_2d_rotation_matrix(self, angle_in_radians: float) -> np.ndarray:
        return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                         [np.sin(angle_in_radians), np.cos(angle_in_radians)]])

    def convert_global_coords_to_local(self, coordinates: np.ndarray, yaw: float) -> np.ndarray:
        transform = self.make_2d_rotation_matrix(angle_in_radians=yaw)
        if len(coordinates.shape) > 2:
            coord_shape = coordinates.shape
            return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
        return np.dot(transform, coordinates.T).T[:, :2]

    def rotate_agents(self, ego_in, ego_out,
                      agents_in, agents_out,
                      roads, agent_types):
        global_translation = []
        use_ego_in = ego_in[-1:, 4]
        use_agents_in = agents_in[:, -1:, 4]

        new_ego_in = np.zeros(
            (ego_in.shape[0], ego_in.shape[1] + 3))  
        new_ego_in[:, 3:] = ego_in 
        new_ego_in[:, 2] = ego_in[:, 4] - use_ego_in  

        new_ego_out = np.zeros((ego_out.shape[0], ego_out.shape[1] + 2))  
        new_ego_out[:, 2:] = ego_out 
        new_ego_out[:, 4] -= use_ego_in  

        new_agents_in = np.zeros((agents_in.shape[0], agents_in.shape[1], agents_in.shape[2] + 3)) 
        new_agents_in[:, :, 3:] = agents_in
        new_agents_in[:, :, 2] = agents_in[:, :, 4] - use_agents_in 

        new_agents_out = np.zeros((agents_out.shape[0], agents_out.shape[1], agents_out.shape[2] + 2))
        new_agents_out[:, :, 2:] = agents_out
        new_agents_out[:, :, 4] -= use_agents_in

        new_roads = roads.copy()

        if agent_types[0, 0]: 
            yaw = ego_in[-1, 4] 
        elif agent_types[0, 1]:  
            # diff = ego_in[-1, :2] - ego_in[-5, :2] 
            diff = ego_in[-1, :2] - ego_in[0, :2]
            yaw = np.arctan2(diff[1], diff[0])

        angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw) 
        translation = ego_in[-1, :2]  
        new_ego_in[:, :2] = self.convert_global_coords_to_local(coordinates=ego_in[:, :2] - translation,
                                                                yaw=angle_of_rotation)
        new_ego_in[:, 5:7] = self.convert_global_coords_to_local(coordinates=ego_in[:, 2:4], yaw=angle_of_rotation)
        new_ego_out[:, :2] = self.convert_global_coords_to_local(coordinates=ego_out[:, :2] - translation,
                                                                 yaw=angle_of_rotation)
        new_roads[0, :, :, :2] = self.convert_global_coords_to_local(coordinates=new_roads[0, :, :, :2] - translation,
                                                                     yaw=angle_of_rotation)
        new_roads[0][np.where(new_roads[0, :, :, -1] == 0)] = 0.0
        global_translation.append([translation[0],translation[1],angle_of_rotation])


        # other agents
        for n in range(self.num_others):
            if not agents_in[n, -1, -1]:
                global_translation.append([0,0,0])
                continue

            if agent_types[n + 1, 0]:  # vehicle
                yaw = agents_in[n, -1, 4]
            elif agent_types[n + 1, 1]:  # pedestrian/bike
                diff = agents_in[n, -1, :2] - agents_in[n, 0, :2]
                yaw = np.arctan2(diff[1], diff[0])
            agent_angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
            agent_translation = agents_in[n, -1, :2]

            use_agent_translation = agent_translation
            use_agents_angle_of_rotation = agent_angle_of_rotation

            new_agents_in[n, :, :2] = self.convert_global_coords_to_local(coordinates=agents_in[n, :, :2] - use_agent_translation,
                                                                          yaw=use_agents_angle_of_rotation)
            new_agents_in[n, :, 5:7] = self.convert_global_coords_to_local(coordinates=agents_in[n, :, 2:4],
                                                                           yaw=use_agents_angle_of_rotation)
            new_agents_out[n, :, :2] = self.convert_global_coords_to_local(
                coordinates=agents_out[n, :, :2] - use_agent_translation, yaw=use_agents_angle_of_rotation)
            new_roads[n + 1, :, :, :2] = self.convert_global_coords_to_local(
                coordinates=new_roads[n + 1, :, :, :2] - use_agent_translation, yaw=use_agents_angle_of_rotation)
            new_roads[n + 1][np.where(new_roads[n + 1, :, :, -1] == 0)] = 0.0
            global_translation.append([use_agent_translation[0],use_agent_translation[1],use_agents_angle_of_rotation])
            
        return new_ego_in, new_ego_out, new_agents_in, new_agents_out, new_roads, np.array(global_translation)

    def _plot_debug(self, ego_in, ego_out, agents_in, agents_out, roads):
        for n in range(self.num_others + 1):
            plt.figure()
            if n == 0:
                plt.scatter(ego_in[:, 0], ego_in[:, 1], color='k')
                plt.scatter(ego_out[:, 0], ego_out[:, 1], color='m')
            else:
                if agents_in[n - 1, -1, -1]:
                    plt.scatter(agents_in[n - 1, :, 0], agents_in[n - 1, :, 1], color='k')
                    plt.scatter(agents_out[n - 1, :, 0], agents_out[n - 1, :, 1], color='m')
            for s in range(roads.shape[1]):
                for p in range(roads.shape[2]):
                    if roads[n, s, p, -1]:
                        plt.scatter(roads[n, s, p, 0], roads[n, s, p, 1], color='g')
            plt.show()
        exit()

    def __getitem__(self, idx: int):
        dataset = h5py.File(os.path.join(self.data_root, self.split_name + '_dataset.hdf5'), 'r')
        agents_data = dataset['agents_trajectories'][idx] # [50,40,7]['x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']
        agent_types = dataset['agents_types'][idx]  # 50,2 [1.0, 0.0] if 'car'

        meta_data = dataset['metas'][idx] # np.array([xmin, ymin, xmax, ymax, datafile_id])
        road_fname_key = dataset['map_paths'][idx][0].decode("utf-8").split("/")[-1]
        roads = self.roads[road_fname_key].copy()#


        roads[:, :, :2] -= np.expand_dims(np.array([meta_data[:2]]), 0) 
        roads_plot = roads[:self.max_num_road_segs].copy()  

        original_roads = np.zeros((self.max_num_road_segs, *roads.shape[1:]))
        original_roads[:len(roads)] = roads
        roads = original_roads.copy()
        ego_in_raw, ego_out_raw, agents_in_raw, agents_out_raw, agent_types = self.split_input_output_normalize(agents_data, meta_data,
                                                                                                agent_types)
        roads_raw = self.copy_agent_roads_across_agents(agents_in_raw, roads) #[9N,78LN,80segN,8]

        # normalize scenes so all agents are going up
        if self.evaluation:
            translations = np.concatenate((ego_in_raw[-1:, :2], agents_in_raw[:, -1, :2]), axis=0)

        ego_in, ego_out, agents_in, agents_out, roads, global_translations = self.rotate_agents(ego_in_raw, ego_out_raw,
                                                                                                agents_in_raw, agents_out_raw,
                                                                                                roads_raw, agent_types)

        if self.evaluation:
            ego_out_model = copy.deepcopy(ego_out)
            agents_out_model = copy.deepcopy(agents_out)
            ego_in = ego_in[1::2]
            agents_in = agents_in[:, 1::2]
            ego_out_model = ego_out_model[1::2]  # putting the original coordinate systems
            agents_out_model = agents_out_model[:, 1::2] # putting the original coordinate systems

            model_ego_in = ego_in.copy()
            model_agents_in = agents_in.copy()
            model_ego_in[:, 3:5] = 0.0
            model_agents_in[:, :, 3:5] = 0.0
            ego_in[:, 0:2] = ego_in[:, 3:5]
            agents_in[:, :, 0:2] = agents_in[:, :, 3:5]
            ego_out[:, 0:2] = ego_out[:, 2:4]  # putting the original coordinate systems
            agents_out[:, :, 0:2] = agents_out[:, :, 2:4]  # putting the original coordinate systems
            return model_ego_in, ego_out, model_agents_in.transpose(1, 0, 2), agents_out.transpose(1, 0, 2), roads, \
                    agent_types, ego_out_model, agents_out_model.transpose(1, 0, 2), ego_in, agents_in.transpose(1, 0, 2), original_roads, translations, global_translations
        else:
            ego_in[:, 3:5] = 0.0
            agents_in[:, :, 3:5] = 0.0
            return ego_in, ego_out, agents_in.transpose(1, 0, 2), agents_out.transpose(1, 0, 2), roads, agent_types, global_translations

    def __len__(self):
        return self.dset_len
