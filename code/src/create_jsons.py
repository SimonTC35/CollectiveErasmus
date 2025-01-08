    def update(frame):
        # Run a simulation step
        data = {}
        for i in range(0, 10000):
            sim.run_step()
            # data["x"].append(sim.q[0].tolist())
            # data["y"].append(sim.q[1].tolist())
            # print(sim.p)
            print(i)
            for j, s in enumerate(sim.p):  #
                try:
                    data["sheep" + str(j + 1)].append([s[0].tolist(), s[1].tolist()])
                except:
                    data["sheep" + str(j + 1)] = []
                    data["sheep" + str(j + 1)].append([s[0].tolist(), s[1].tolist()])

            try:
                data["dog" + str(1)].append([sim.q[0].tolist(), sim.q[1].tolist()])
            except:
                data["dog" + str(1)] = []
                data["dog" + str(1)].append([sim.q[0].tolist(), sim.q[1].tolist()])
            # print(sim.q)
            # print(sim.p)
        print(data)
        # exit(0)
        # print(data)
        # Serializing json

        coords = data["dog1"]
        current_entity = {"x": [], "y": []}
        for c in coords:
            current_entity["x"].append(c[0])
            current_entity["y"].append(c[1])
        json_object = json.dumps(current_entity, indent=2)

        # Writing to sample.json
        with open("C:/Users/lukas/Collective Behaviour/Assets/jsons/dog1" + ".json", "w") as outfile:
            outfile.write(json_object)
        i += 1

        i = 1
        while True:
            name = "sheep" + str(i)
            if name not in data:
                break
            coords = data[name]
            current_entity = {"x": [], "y": []}
            for c in coords:
                current_entity["x"].append(c[0])
                current_entity["y"].append(c[1])
            json_object = json.dumps(current_entity, indent=2)

            # Writing to sample.json
            with open("C:/Users/lukas/Collective Behaviour/Assets/jsons/" + name + ".json", "w") as outfile:
                outfile.write(json_object)
            i += 1
        # exit(0)

        # Update sheep positions
        sheep_scatter.set_offsets(sim.p)

        # Update dog position
        dog_marker.set_data([sim.q[0]], [sim.q[1]])

        # Update the step text
        step_text.set_text(f'Step: {frame}')
        return sheep_scatter, dog_marker