import os
import os.path as osp
from fhirpy import SyncFHIRClient

## output dir
here = osp.dirname(osp.abspath(__file__))
out_dir = osp.join(here, 'output')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

## output file
if not osp.exists(osp.join(out_dir, 'report.txt')):
    with open(osp.join(out_dir, 'report.txt'), 'w') as f:
        pass

## Define (input) variables from Docker Container environment variables
fhir_server = str(os.environ['FHIR_SERVER'])
fhir_port = str(os.environ['FHIR_PORT'])

print('http://{}:{}/fhir'.format(fhir_server, fhir_port))

# Create an instance
client = SyncFHIRClient('http://{}:{}/fhir'.format(fhir_server, fhir_port))
resources = ['Patient', 'Condition', 'Observation', 'Specimen']
# Search for Resource
with open(osp.join(out_dir, 'report.txt'), 'a') as f:
    for resource in resources:
        count = 0
        items = client.resources(resource)  # Return lazy search set
        for item in items:
            count = count + 1
        print("Number of '{}': {}".format(resource, count))    
        f.write("Number of '{}': {} \n".format(resource, count))
print("Done")
